from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    CONTENT_POLICY_RAW_USER_TEXT_ALLOWED,
    CONTENT_POLICY_REDACTED,
    DIAGNOSTIC_CATEGORY_LIFECYCLE,
    DIAGNOSTIC_CATEGORY_UNKNOWN,
    DIAGNOSTIC_FIELD_KEY_MAX_LENGTH,
    DIAGNOSTIC_FIELD_MAX_ITEMS,
    DIAGNOSTIC_FIELD_VALUE_MAX_LENGTH,
    DIAGNOSTIC_VISIBILITY_BASIC,
    DIAGNOSTIC_VISIBILITY_DETAILED,
    DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
    DIAGNOSTIC_VISIBILITY_PERSISTED_FAILURE_ONLY,
    ContentPolicy,
    DiagnosticVisibility,
    ErrorDiagnostics,
    SafeMessageParam,
    UserMessageRef,
)

DiagnosticSink: TypeAlias = Literal[
    "dashboard",
    "snackbar",
    "chatbox_disclosure",
    "basic_logs",
    "detailed_logs",
    "persisted_logs",
    "failure_jsonl",
]
DIAGNOSTIC_SINK_DASHBOARD: Final[DiagnosticSink] = "dashboard"
DIAGNOSTIC_SINK_SNACKBAR: Final[DiagnosticSink] = "snackbar"
DIAGNOSTIC_SINK_CHATBOX_DISCLOSURE: Final[DiagnosticSink] = "chatbox_disclosure"
DIAGNOSTIC_SINK_BASIC_LOGS: Final[DiagnosticSink] = "basic_logs"
DIAGNOSTIC_SINK_DETAILED_LOGS: Final[DiagnosticSink] = "detailed_logs"
DIAGNOSTIC_SINK_PERSISTED_LOGS: Final[DiagnosticSink] = "persisted_logs"
DIAGNOSTIC_SINK_FAILURE_JSONL: Final[DiagnosticSink] = "failure_jsonl"
DIAGNOSTIC_SINKS: Final[tuple[DiagnosticSink, ...]] = (
    DIAGNOSTIC_SINK_DASHBOARD,
    DIAGNOSTIC_SINK_SNACKBAR,
    DIAGNOSTIC_SINK_CHATBOX_DISCLOSURE,
    DIAGNOSTIC_SINK_BASIC_LOGS,
    DIAGNOSTIC_SINK_DETAILED_LOGS,
    DIAGNOSTIC_SINK_PERSISTED_LOGS,
    DIAGNOSTIC_SINK_FAILURE_JSONL,
)

DIAGNOSTIC_SINK_VISIBILITY_RULES: Final[
    Mapping[DiagnosticSink, frozenset[DiagnosticVisibility]]
] = MappingProxyType(
    {
        DIAGNOSTIC_SINK_DASHBOARD: frozenset({DIAGNOSTIC_VISIBILITY_BASIC}),
        DIAGNOSTIC_SINK_SNACKBAR: frozenset({DIAGNOSTIC_VISIBILITY_BASIC}),
        DIAGNOSTIC_SINK_CHATBOX_DISCLOSURE: frozenset({DIAGNOSTIC_VISIBILITY_BASIC}),
        DIAGNOSTIC_SINK_BASIC_LOGS: frozenset({DIAGNOSTIC_VISIBILITY_BASIC}),
        DIAGNOSTIC_SINK_DETAILED_LOGS: frozenset(
            {
                DIAGNOSTIC_VISIBILITY_BASIC,
                DIAGNOSTIC_VISIBILITY_DETAILED,
                DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
            }
        ),
        DIAGNOSTIC_SINK_PERSISTED_LOGS: frozenset(
            {
                DIAGNOSTIC_VISIBILITY_BASIC,
                DIAGNOSTIC_VISIBILITY_DETAILED,
                DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
            }
        ),
        DIAGNOSTIC_SINK_FAILURE_JSONL: frozenset({DIAGNOSTIC_VISIBILITY_PERSISTED_FAILURE_ONLY}),
    }
)

DiagnosticValidationStatus: TypeAlias = Literal["accepted", "rejected"]
DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED: Final[DiagnosticValidationStatus] = "accepted"
DIAGNOSTIC_VALIDATION_STATUS_REJECTED: Final[DiagnosticValidationStatus] = "rejected"

DiagnosticValidationReason: TypeAlias = Literal[
    "visibility_forbidden",
    "unsupported_field_type",
    "field_limit_exceeded",
    "excessive_depth",
    "secret_pattern",
    "broker_raw_message",
    "provider_response_body",
    "sensitive_local_llm_extra_body",
    "unsafe_text_payload",
]
DIAGNOSTIC_VALIDATION_REASON_VISIBILITY_FORBIDDEN: Final[DiagnosticValidationReason] = (
    "visibility_forbidden"
)
DIAGNOSTIC_VALIDATION_REASON_UNSUPPORTED_FIELD_TYPE: Final[DiagnosticValidationReason] = (
    "unsupported_field_type"
)
DIAGNOSTIC_VALIDATION_REASON_FIELD_LIMIT_EXCEEDED: Final[DiagnosticValidationReason] = (
    "field_limit_exceeded"
)
DIAGNOSTIC_VALIDATION_REASON_EXCESSIVE_DEPTH: Final[DiagnosticValidationReason] = "excessive_depth"
DIAGNOSTIC_VALIDATION_REASON_SECRET_PATTERN: Final[DiagnosticValidationReason] = "secret_pattern"
DIAGNOSTIC_VALIDATION_REASON_BROKER_RAW_MESSAGE: Final[DiagnosticValidationReason] = (
    "broker_raw_message"
)
DIAGNOSTIC_VALIDATION_REASON_PROVIDER_RESPONSE_BODY: Final[DiagnosticValidationReason] = (
    "provider_response_body"
)
DIAGNOSTIC_VALIDATION_REASON_SENSITIVE_LOCAL_LLM_EXTRA_BODY: Final[DiagnosticValidationReason] = (
    "sensitive_local_llm_extra_body"
)
DIAGNOSTIC_VALIDATION_REASON_UNSAFE_TEXT_PAYLOAD: Final[DiagnosticValidationReason] = (
    "unsafe_text_payload"
)

DIAGNOSTIC_FIELD_MAX_DEPTH: Final = 3
DIAGNOSTIC_REDACTION_MARKER: Final = "[redacted]"
BROKER_RAW_MESSAGE_REDACTION_MARKER: Final = "[broker-raw-message-redacted]"
PROVIDER_RESPONSE_BODY_REDACTION_MARKER: Final = "[provider-response-body-redacted]"
LOCAL_LLM_EXTRA_BODY_REDACTION_MARKER: Final = "[local-llm-extra-body-redacted]"
DESKTOP_RENDERER_EVENT_SCHEMA_VERSION: Final = 1
DESKTOP_OVERLAY_REPRO_SCHEMA_VERSION: Final = 1
_DESKTOP_RENDERER_EVENT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "record_type",
        "event_type",
        "renderer_revision",
        "actual_disposition",
        "render_commit_acknowledged",
        "slot_count",
        "line_count",
        "surface_visible",
        "interaction_mode",
        "window_width",
        "window_height",
    }
)
_DESKTOP_RENDERER_EVENT_DISPOSITIONS: Final = MappingProxyType(
    {
        "receipt": frozenset({"committed", "superseded", "stale", "failed"}),
        "supersession": frozenset({"superseded"}),
        "stale": frozenset({"stale"}),
        "failed": frozenset({"failed"}),
        "render_start": frozenset({"committed"}),
        "render_commit": frozenset({"committed"}),
        "render_commit_acknowledgement": frozenset({"committed"}),
    }
)
_DESKTOP_OVERLAY_REPRO_DISPOSITIONS: Final = frozenset(
    {"committed", "superseded", "stale", "failed"}
)
_DESKTOP_OVERLAY_REPRO_RECORD_FIELDS: Final = frozenset(
    {
        "schema_version",
        "record_type",
        "cycle",
        "wall_clock_utc",
        "monotonic_ms",
        "synthetic_revision",
        "expected_disposition",
        "actual_disposition",
        "render_commit_acknowledged",
        "slot_count",
        "line_count",
        "surface_visible",
        "interaction_mode",
        "window_width",
        "window_height",
    }
)
_DESKTOP_OVERLAY_REPRO_RESULT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "record_type",
        "outcome",
        "reason",
        "cycles_requested",
        "cycles_completed",
        "committed_count",
        "superseded_count",
        "stale_count",
        "failed_count",
        "renderer_shutdown_completed",
        "bridge_shutdown_completed",
        "backdrop_shutdown_completed",
    }
)
_DESKTOP_OVERLAY_REPRO_FAILURE_REASONS: Final = frozenset(
    {
        "invalid_argument",
        "startup_failed",
        "bridge_failed",
        "render_failed",
        "render_timeout",
        "validation_failed",
        "cleanup_failed",
        "artifact_invalid",
    }
)

_SAFE_REDACTION_MARKERS: Final = frozenset(
    {
        BROKER_RAW_MESSAGE_REDACTION_MARKER,
        DIAGNOSTIC_REDACTION_MARKER,
        PROVIDER_RESPONSE_BODY_REDACTION_MARKER,
        LOCAL_LLM_EXTRA_BODY_REDACTION_MARKER,
    }
)
_SENSITIVE_KEY_FRAGMENTS: Final = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "managed_private_key",
        "password",
        "private_key",
        "secret",
        "session_token",
        "token",
    }
)
_SENSITIVE_TOKEN_KEYS: Final = frozenset(
    {
        "access_token",
        "auth_token",
        "bearer_token",
        "id_token",
        "refresh_token",
        "session_token",
        "token",
    }
)
_LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS: Final = frozenset(
    {"api_key", "authorization", "headers", "password", "secret", "token"}
)
BROKER_RAW_MESSAGE_KEYS: Final = frozenset(
    {
        "broker_eligibility_message",
        "broker_raw_eligibility_message",
        "broker_raw_message",
        "broker_sensitive_message",
        "raw_broker_eligibility_message",
        "raw_broker_message",
    }
)
PROVIDER_RESPONSE_BODY_KEYS: Final = frozenset(
    {
        "provider_payload",
        "provider_response",
        "provider_response_body",
        "raw_body",
        "raw_payload",
        "raw_provider_payload",
        "raw_provider_response",
        "raw_response",
        "raw_response_body",
        "response_body",
        "response_text",
    }
)
_BROKER_RAW_MESSAGE_KEYS: Final = BROKER_RAW_MESSAGE_KEYS
_PROVIDER_RESPONSE_BODY_KEYS: Final = PROVIDER_RESPONSE_BODY_KEYS
_UNSAFE_TEXT_KEYS: Final = frozenset(
    {
        "exception_text",
        "file_content",
        "file_contents",
        "raw_exception",
        "raw_file",
        "raw_source_text",
        "raw_transcript",
        "raw_translation",
        "source_text",
        "stack_trace",
        "traceback",
        "transcript_text",
        "translated_text",
        "translation_text",
    }
)
_SECRET_VALUE_PATTERNS: Final = (
    re.compile(r"(?i)\bmanaged[\s_-]?private[\s_-]?key\b\s*[\"']?\s*[:=]"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|authorization|password|private[_-]?key|secret|session[_-]?token|token)\b[\"']?\s*[:=]\s*[\"']?[^\s\"',;}]+"
    ),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+\-/]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9._-]{8,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
_STACK_TRACE_PATTERNS: Final = (
    re.compile(r"Traceback \(most recent call last\):"),
    re.compile(r"Stack \(most recent call last\):"),
    re.compile(r'File "[^"]+", line \d+'),
)
_PRIVATE_KEY_BLOCK_TEXT_RE: Final = re.compile(
    r"(?is)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----"
)
_CAMEL_CASE_BOUNDARY_RE: Final = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SECRET_ASSIGNMENT_TEXT_RE: Final = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|private[_-]?key|secret|session[_-]?token|token)\b[\"']?\s*[:=]\s*[\"']?(?:Bearer\s+[A-Za-z0-9._~+\-/]{8,}|[^\s\"',;}]+)"
)
_BEARER_SECRET_TEXT_RE: Final = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+\-/]{8,}")
_OPENAI_STYLE_SECRET_TEXT_RE: Final = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9._-]{8,}\b")
_URL_USERINFO_RE: Final = re.compile(
    r"(?i)\b([a-z][a-z0-9+.-]*://)([^/\s:@]+):([^/\s@]+)@"
)
CONVERSATION_TEXT_MAX_LENGTH: Final = 4096


def _raw_text_key_assignment_re(keys: frozenset[str]) -> re.Pattern[str]:
    alternatives = sorted(
        (_raw_text_key_pattern(key) for key in keys),
        key=len,
        reverse=True,
    )
    return re.compile(r"(?is)\b(?:" + "|".join(alternatives) + r")\b[\"']?\s*[=:]\s*")


def _raw_text_key_pattern(key: str) -> str:
    return r"[_ -]?".join(re.escape(part) for part in key.split("_"))


_PROVIDER_RESPONSE_BODY_TEXT_RE: Final = _raw_text_key_assignment_re(_PROVIDER_RESPONSE_BODY_KEYS)
_BROKER_RAW_MESSAGE_TEXT_RE: Final = _raw_text_key_assignment_re(_BROKER_RAW_MESSAGE_KEYS)
_UNSAFE_TEXT_PAYLOAD_TEXT_RE: Final = _raw_text_key_assignment_re(_UNSAFE_TEXT_KEYS)
_SENSITIVE_TOKEN_ASSIGNMENT_TEXT_RE: Final = _raw_text_key_assignment_re(_SENSITIVE_TOKEN_KEYS)
_LOCAL_LLM_EXTRA_BODY_TEXT_RE: Final = re.compile(
    r"(?is)\b(?:local[_ -]?(?:llm|openai)[_ -]?extra[_ -]?body)\b\s*[=:]\s*"
)


@dataclass(frozen=True, slots=True)
class DiagnosticRedactionPolicy:
    allow_sensitive_local_llm_extra_body_fields: bool = False


DEFAULT_DIAGNOSTIC_REDACTION_POLICY: Final = DiagnosticRedactionPolicy()
MESSAGE_PARAM_REDACTION_POLICY: Final = DiagnosticRedactionPolicy(
    allow_sensitive_local_llm_extra_body_fields=True,
)


@dataclass(frozen=True, slots=True)
class DiagnosticValidationResult:
    status: DiagnosticValidationStatus
    sink: DiagnosticSink
    diagnostics: ErrorDiagnostics | None
    redacted: bool
    reasons: tuple[DiagnosticValidationReason, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))


@dataclass(frozen=True, slots=True)
class DiagnosticTextValidationResult:
    status: DiagnosticValidationStatus
    sink: DiagnosticSink
    text: str | None
    redacted: bool
    reasons: tuple[DiagnosticValidationReason, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))


def validate_diagnostics_for_sink(
    diagnostics: ErrorDiagnostics,
    sink: DiagnosticSink,
    *,
    policy: DiagnosticRedactionPolicy = DEFAULT_DIAGNOSTIC_REDACTION_POLICY,
) -> DiagnosticValidationResult:
    reasons = _validation_reasons(diagnostics, sink, policy=policy)
    if reasons:
        return DiagnosticValidationResult(
            status=DIAGNOSTIC_VALIDATION_STATUS_REJECTED,
            sink=sink,
            diagnostics=None,
            redacted=False,
            reasons=reasons,
        )
    return DiagnosticValidationResult(
        status=DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED,
        sink=sink,
        diagnostics=diagnostics,
        redacted=False,
        reasons=(),
    )


def validate_desktop_renderer_event(
    record: Mapping[str, object],
) -> Mapping[str, object] | None:
    candidate = dict(record)
    if set(candidate) != _DESKTOP_RENDERER_EVENT_FIELDS:
        return None
    schema_version = candidate.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != DESKTOP_RENDERER_EVENT_SCHEMA_VERSION
    ):
        return None
    if candidate.get("record_type") != "renderer_event":
        return None
    event_type = candidate.get("event_type")
    disposition = candidate.get("actual_disposition")
    if (
        not isinstance(event_type, str)
        or not isinstance(disposition, str)
        or disposition not in _DESKTOP_RENDERER_EVENT_DISPOSITIONS.get(event_type, frozenset())
    ):
        return None
    if not _is_nonnegative_int(candidate.get("renderer_revision")):
        return None
    if not _is_bounded_int(candidate.get("slot_count"), minimum=0, maximum=2):
        return None
    if not _is_bounded_int(candidate.get("line_count"), minimum=0, maximum=6):
        return None
    if not isinstance(candidate.get("surface_visible"), bool):
        return None
    if candidate.get("interaction_mode") not in {"edit", "locked"}:
        return None
    if not _is_bounded_int(candidate.get("window_width"), minimum=1):
        return None
    if not _is_bounded_int(candidate.get("window_height"), minimum=1):
        return None
    acknowledged = candidate.get("render_commit_acknowledged")
    if not isinstance(acknowledged, bool):
        return None
    if event_type == "render_commit_acknowledgement":
        if not acknowledged:
            return None
    elif acknowledged:
        return None
    validation = validate_diagnostics_for_sink(
        ErrorDiagnostics(
            component="desktop_overlay_renderer",
            operation="snapshot_scheduling",
            code=event_type,
            category=DIAGNOSTIC_CATEGORY_LIFECYCLE,
            visibility=DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
            content_policy=CONTENT_POLICY_METADATA_ONLY,
            status_code=None,
            retry_after_ms=None,
            fields=candidate,
        ),
        DIAGNOSTIC_SINK_PERSISTED_LOGS,
    )
    if validation.status != DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED:
        return None
    return MappingProxyType(candidate)


def validate_desktop_overlay_repro_record(
    record: Mapping[str, object],
) -> Mapping[str, object] | None:
    candidate = dict(record)
    if set(candidate) != _DESKTOP_OVERLAY_REPRO_RECORD_FIELDS:
        return None
    if not _is_exact_int(candidate.get("schema_version"), DESKTOP_OVERLAY_REPRO_SCHEMA_VERSION):
        return None
    if candidate.get("record_type") != "revision_outcome" or not isinstance(
        candidate.get("record_type"), str
    ):
        return None
    if not _is_bounded_int(candidate.get("cycle"), minimum=1):
        return None
    wall_clock = candidate.get("wall_clock_utc")
    if not _is_rfc3339_utc(wall_clock):
        return None
    if not _is_nonnegative_int(candidate.get("monotonic_ms")) or not _is_nonnegative_int(
        candidate.get("synthetic_revision")
    ):
        return None
    expected = candidate.get("expected_disposition")
    actual = candidate.get("actual_disposition")
    if (
        not isinstance(expected, str)
        or not isinstance(actual, str)
        or expected not in _DESKTOP_OVERLAY_REPRO_DISPOSITIONS
        or actual not in _DESKTOP_OVERLAY_REPRO_DISPOSITIONS
    ):
        return None
    acknowledged = candidate.get("render_commit_acknowledged")
    if not isinstance(acknowledged, bool) or acknowledged != (actual == "committed"):
        return None
    if not _valid_desktop_overlay_repro_visual_state(candidate):
        return None
    return _validate_desktop_overlay_repro_candidate(candidate, "revision_outcome")


def validate_desktop_overlay_repro_result(
    result: Mapping[str, object],
) -> Mapping[str, object] | None:
    candidate = dict(result)
    if set(candidate) != _DESKTOP_OVERLAY_REPRO_RESULT_FIELDS:
        return None
    if not _is_exact_int(candidate.get("schema_version"), DESKTOP_OVERLAY_REPRO_SCHEMA_VERSION):
        return None
    if candidate.get("record_type") != "run_result" or not isinstance(
        candidate.get("record_type"), str
    ):
        return None
    outcome = candidate.get("outcome")
    reason = candidate.get("reason")
    if not isinstance(outcome, str) or outcome not in {"completed", "failed"}:
        return None
    if outcome == "completed" and reason is not None:
        return None
    if outcome == "failed" and (
        not isinstance(reason, str) or reason not in _DESKTOP_OVERLAY_REPRO_FAILURE_REASONS
    ):
        return None
    for key in (
        "cycles_requested",
        "cycles_completed",
        "committed_count",
        "superseded_count",
        "stale_count",
        "failed_count",
    ):
        if not _is_nonnegative_int(candidate.get(key)):
            return None
    if candidate["cycles_completed"] > candidate["cycles_requested"]:
        return None
    if not all(
        isinstance(candidate.get(key), bool)
        for key in (
            "renderer_shutdown_completed",
            "bridge_shutdown_completed",
            "backdrop_shutdown_completed",
        )
    ):
        return None
    return _validate_desktop_overlay_repro_candidate(candidate, "run_result")


def _valid_desktop_overlay_repro_visual_state(candidate: Mapping[str, object]) -> bool:
    return (
        _is_bounded_int(candidate.get("slot_count"), minimum=0, maximum=2)
        and _is_bounded_int(candidate.get("line_count"), minimum=0, maximum=6)
        and isinstance(candidate.get("surface_visible"), bool)
        and candidate.get("interaction_mode") in {"edit", "locked"}
        and _is_bounded_int(candidate.get("window_width"), minimum=1)
        and _is_bounded_int(candidate.get("window_height"), minimum=1)
    )


def _validate_desktop_overlay_repro_candidate(
    candidate: dict[str, object], operation: str
) -> Mapping[str, object] | None:
    validation = validate_diagnostics_for_sink(
        ErrorDiagnostics(
            component="desktop_overlay_repro",
            operation=operation,
            code="v1",
            category=DIAGNOSTIC_CATEGORY_LIFECYCLE,
            visibility=DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
            content_policy=CONTENT_POLICY_METADATA_ONLY,
            status_code=None,
            retry_after_ms=None,
            fields=candidate,
        ),
        DIAGNOSTIC_SINK_PERSISTED_LOGS,
    )
    if validation.status != DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED:
        return None
    return MappingProxyType(candidate)


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_exact_int(value: object, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _is_rfc3339_utc(value: object) -> bool:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", value
    ):
        return False
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)


def _is_bounded_int(value: object, *, minimum: int, maximum: int | None = None) -> bool:
    if not _is_nonnegative_int(value) or value < minimum:
        return False
    return maximum is None or value <= maximum


def redact_diagnostics_for_sink(
    diagnostics: ErrorDiagnostics,
    sink: DiagnosticSink,
    *,
    policy: DiagnosticRedactionPolicy = DEFAULT_DIAGNOSTIC_REDACTION_POLICY,
) -> DiagnosticValidationResult:
    fields, redacted = _redact_fields(diagnostics, policy=policy)
    redacted_diagnostics = _copy_diagnostics_with_fields(diagnostics, fields)
    reasons = _validation_reasons(redacted_diagnostics, sink, policy=policy)
    if reasons:
        return DiagnosticValidationResult(
            status=DIAGNOSTIC_VALIDATION_STATUS_REJECTED,
            sink=sink,
            diagnostics=None,
            redacted=redacted,
            reasons=reasons,
        )
    return DiagnosticValidationResult(
        status=DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED,
        sink=sink,
        diagnostics=redacted_diagnostics,
        redacted=redacted,
        reasons=(),
    )


def redact_text_for_sink(text: str, sink: DiagnosticSink) -> DiagnosticTextValidationResult:
    redacted_text, redacted = _redact_text_payload(text)
    reasons = _text_content_reasons(redacted_text)
    if reasons:
        return DiagnosticTextValidationResult(
            status=DIAGNOSTIC_VALIDATION_STATUS_REJECTED,
            sink=sink,
            text=None,
            redacted=redacted,
            reasons=reasons,
        )
    return DiagnosticTextValidationResult(
        status=DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED,
        sink=sink,
        text=redacted_text,
        redacted=redacted,
        reasons=(),
    )


def redact_conversation_text_for_sink(
    text: str,
    sink: DiagnosticSink,
) -> DiagnosticTextValidationResult:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    truncated = len(normalized) > CONVERSATION_TEXT_MAX_LENGTH
    if truncated:
        normalized = f"{normalized[:CONVERSATION_TEXT_MAX_LENGTH]}…[truncated]"
    redacted, changed = _redact_conversation_payload(normalized)
    return DiagnosticTextValidationResult(
        status=DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED,
        sink=sink,
        text=redacted,
        redacted=changed or truncated,
        reasons=(),
    )


def redact_message_params_for_sink(
    params: Mapping[str, SafeMessageParam],
    sink: DiagnosticSink,
) -> Mapping[str, SafeMessageParam]:
    diagnostics = ErrorDiagnostics(
        component="user_message",
        operation="localize",
        code="user_message.params",
        category=DIAGNOSTIC_CATEGORY_UNKNOWN,
        visibility=_compatible_visibility_for_sink(sink),
        content_policy=CONTENT_POLICY_REDACTED,
        status_code=None,
        retry_after_ms=None,
        fields=params,
    )
    validation = redact_diagnostics_for_sink(
        diagnostics,
        sink,
        policy=MESSAGE_PARAM_REDACTION_POLICY,
    )
    if validation.status != DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED:
        return MappingProxyType({})
    if validation.diagnostics is None:
        return MappingProxyType({})
    return MappingProxyType(
        {
            key: value
            for key, value in validation.diagnostics.fields.items()
            if _is_supported_diagnostic_value(value)
        }
    )


def redact_user_message_ref_for_sink(
    message: UserMessageRef,
    sink: DiagnosticSink,
) -> UserMessageRef:
    return UserMessageRef(
        key=message.key,
        params=redact_message_params_for_sink(message.params, sink),
        severity=message.severity,
    )


def _validation_reasons(
    diagnostics: ErrorDiagnostics,
    sink: DiagnosticSink,
    *,
    policy: DiagnosticRedactionPolicy,
) -> tuple[DiagnosticValidationReason, ...]:
    reasons: list[DiagnosticValidationReason] = []
    allowed_visibility = DIAGNOSTIC_SINK_VISIBILITY_RULES.get(sink)
    if allowed_visibility is None or diagnostics.visibility not in allowed_visibility:
        reasons.append(DIAGNOSTIC_VALIDATION_REASON_VISIBILITY_FORBIDDEN)

    reasons.extend(_field_shape_reasons(diagnostics.fields))
    reasons.extend(
        _content_reasons(
            diagnostics.fields,
            content_policy=diagnostics.content_policy,
            policy=policy,
        )
    )
    return tuple(dict.fromkeys(reasons))


def _compatible_visibility_for_sink(sink: DiagnosticSink) -> DiagnosticVisibility:
    allowed = DIAGNOSTIC_SINK_VISIBILITY_RULES.get(sink, frozenset())
    for visibility in (
        DIAGNOSTIC_VISIBILITY_BASIC,
        DIAGNOSTIC_VISIBILITY_DETAILED,
        DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
        DIAGNOSTIC_VISIBILITY_PERSISTED_FAILURE_ONLY,
    ):
        if visibility in allowed:
            return visibility
    return DIAGNOSTIC_VISIBILITY_BASIC


def _field_shape_reasons(
    fields: Mapping[str, object],
) -> tuple[DiagnosticValidationReason, ...]:
    reasons: list[DiagnosticValidationReason] = []
    if len(fields) > DIAGNOSTIC_FIELD_MAX_ITEMS:
        reasons.append(DIAGNOSTIC_VALIDATION_REASON_FIELD_LIMIT_EXCEEDED)

    for key, value in fields.items():
        if not isinstance(key, str):
            reasons.append(DIAGNOSTIC_VALIDATION_REASON_UNSUPPORTED_FIELD_TYPE)
            continue
        if len(key) > DIAGNOSTIC_FIELD_KEY_MAX_LENGTH:
            reasons.append(DIAGNOSTIC_VALIDATION_REASON_FIELD_LIMIT_EXCEEDED)
        if _max_depth(value) > DIAGNOSTIC_FIELD_MAX_DEPTH:
            reasons.append(DIAGNOSTIC_VALIDATION_REASON_EXCESSIVE_DEPTH)
        if not _is_supported_diagnostic_value(value):
            reasons.append(DIAGNOSTIC_VALIDATION_REASON_UNSUPPORTED_FIELD_TYPE)
            continue
        if isinstance(value, str) and len(value) > DIAGNOSTIC_FIELD_VALUE_MAX_LENGTH:
            reasons.append(DIAGNOSTIC_VALIDATION_REASON_FIELD_LIMIT_EXCEEDED)
        if isinstance(value, float) and not math.isfinite(value):
            reasons.append(DIAGNOSTIC_VALIDATION_REASON_UNSUPPORTED_FIELD_TYPE)

    return tuple(dict.fromkeys(reasons))


def _content_reasons(
    fields: Mapping[str, object],
    *,
    content_policy: ContentPolicy,
    policy: DiagnosticRedactionPolicy,
) -> tuple[DiagnosticValidationReason, ...]:
    reasons: list[DiagnosticValidationReason] = []
    for key, value in fields.items():
        key_text = str(key)
        if _contains_unredacted_provider_response_body(key_text, value):
            reasons.append(DIAGNOSTIC_VALIDATION_REASON_PROVIDER_RESPONSE_BODY)
        if _contains_unredacted_broker_raw_message(key_text, value):
            reasons.append(DIAGNOSTIC_VALIDATION_REASON_BROKER_RAW_MESSAGE)
        if _contains_unredacted_sensitive_local_llm_extra_body(key_text, value):
            if not policy.allow_sensitive_local_llm_extra_body_fields:
                reasons.append(DIAGNOSTIC_VALIDATION_REASON_SENSITIVE_LOCAL_LLM_EXTRA_BODY)
            elif value != LOCAL_LLM_EXTRA_BODY_REDACTION_MARKER:
                reasons.append(DIAGNOSTIC_VALIDATION_REASON_SENSITIVE_LOCAL_LLM_EXTRA_BODY)
        if _contains_unredacted_secret_pattern(key_text, value):
            reasons.append(DIAGNOSTIC_VALIDATION_REASON_SECRET_PATTERN)
        if (
            content_policy != CONTENT_POLICY_RAW_USER_TEXT_ALLOWED
            and _contains_unredacted_unsafe_text_payload(key_text, value)
        ):
            reasons.append(DIAGNOSTIC_VALIDATION_REASON_UNSAFE_TEXT_PAYLOAD)
    return tuple(dict.fromkeys(reasons))


def _redact_fields(
    diagnostics: ErrorDiagnostics,
    *,
    policy: DiagnosticRedactionPolicy,
) -> tuple[Mapping[str, object], bool]:
    redacted = False
    fields: dict[str, object] = {}
    for key, value in diagnostics.fields.items():
        key_text = str(key)
        if _contains_unredacted_provider_response_body(key_text, value):
            if diagnostics.content_policy == CONTENT_POLICY_REDACTED:
                fields[key] = PROVIDER_RESPONSE_BODY_REDACTION_MARKER
                redacted = True
            else:
                fields[key] = value
            continue

        if _contains_unredacted_broker_raw_message(key_text, value):
            if diagnostics.content_policy == CONTENT_POLICY_REDACTED:
                fields[key] = BROKER_RAW_MESSAGE_REDACTION_MARKER
                redacted = True
            else:
                fields[key] = value
            continue

        if _contains_unredacted_sensitive_local_llm_extra_body(key_text, value):
            if (
                diagnostics.content_policy == CONTENT_POLICY_REDACTED
                and policy.allow_sensitive_local_llm_extra_body_fields
            ):
                fields[key] = LOCAL_LLM_EXTRA_BODY_REDACTION_MARKER
                redacted = True
            else:
                fields[key] = value
            continue

        if _contains_unredacted_secret_pattern(key_text, value):
            if diagnostics.content_policy == CONTENT_POLICY_REDACTED:
                fields[key] = DIAGNOSTIC_REDACTION_MARKER
                redacted = True
            else:
                fields[key] = value
            continue

        if _contains_unredacted_unsafe_text_payload(key_text, value):
            if diagnostics.content_policy == CONTENT_POLICY_REDACTED:
                fields[key] = DIAGNOSTIC_REDACTION_MARKER
                redacted = True
            else:
                fields[key] = value
            continue

        fields[key] = value
    return MappingProxyType(fields), redacted


def _redact_text_payload(text: str) -> tuple[str, bool]:
    redacted = text.replace("\r\n", "\n").replace("\r", "\n")
    redacted = _redact_raw_assignment_values(
        redacted,
        _PROVIDER_RESPONSE_BODY_TEXT_RE,
        PROVIDER_RESPONSE_BODY_REDACTION_MARKER,
    )
    redacted = _redact_raw_assignment_values(
        redacted,
        _BROKER_RAW_MESSAGE_TEXT_RE,
        BROKER_RAW_MESSAGE_REDACTION_MARKER,
    )
    redacted = _redact_raw_assignment_values(
        redacted,
        _LOCAL_LLM_EXTRA_BODY_TEXT_RE,
        LOCAL_LLM_EXTRA_BODY_REDACTION_MARKER,
    )
    redacted = _redact_raw_assignment_values(
        redacted,
        _UNSAFE_TEXT_PAYLOAD_TEXT_RE,
        DIAGNOSTIC_REDACTION_MARKER,
    )
    redacted = _redact_raw_assignment_values(
        redacted,
        _SENSITIVE_TOKEN_ASSIGNMENT_TEXT_RE,
        DIAGNOSTIC_REDACTION_MARKER,
    )
    redacted = re.sub(
        r"(?is)\n?Traceback \(most recent call last\):.*",
        DIAGNOSTIC_REDACTION_MARKER,
        redacted,
    )
    redacted = re.sub(
        r"(?is)\n?Stack \(most recent call last\):.*",
        DIAGNOSTIC_REDACTION_MARKER,
        redacted,
    )
    redacted = re.sub(r'(?im)^\s*File "[^"]+", line \d+.*$', DIAGNOSTIC_REDACTION_MARKER, redacted)
    redacted = _PRIVATE_KEY_BLOCK_TEXT_RE.sub(DIAGNOSTIC_REDACTION_MARKER, redacted)
    redacted = _SECRET_ASSIGNMENT_TEXT_RE.sub(_redact_secret_assignment, redacted)
    redacted = _BEARER_SECRET_TEXT_RE.sub(f"Bearer {DIAGNOSTIC_REDACTION_MARKER}", redacted)
    redacted = _OPENAI_STYLE_SECRET_TEXT_RE.sub(DIAGNOSTIC_REDACTION_MARKER, redacted)
    redacted = _URL_USERINFO_RE.sub(
        lambda match: f"{match.group(1)}[redacted]@",
        redacted,
    )
    changed = redacted != text
    if changed:
        redacted = re.sub(r"\s+", " ", redacted).strip()
    return redacted or DIAGNOSTIC_REDACTION_MARKER, changed


def _redact_conversation_payload(text: str) -> tuple[str, bool]:
    redacted = _PRIVATE_KEY_BLOCK_TEXT_RE.sub(DIAGNOSTIC_REDACTION_MARKER, text)
    redacted = _SECRET_ASSIGNMENT_TEXT_RE.sub(_redact_secret_assignment, redacted)
    redacted = _BEARER_SECRET_TEXT_RE.sub(f"Bearer {DIAGNOSTIC_REDACTION_MARKER}", redacted)
    redacted = _OPENAI_STYLE_SECRET_TEXT_RE.sub(DIAGNOSTIC_REDACTION_MARKER, redacted)
    redacted = _URL_USERINFO_RE.sub(
        lambda match: f"{match.group(1)}[redacted]@",
        redacted,
    )
    changed = redacted != text
    return redacted or DIAGNOSTIC_REDACTION_MARKER, changed


def _text_content_reasons(text: str) -> tuple[DiagnosticValidationReason, ...]:
    reasons: list[DiagnosticValidationReason] = []
    if _PROVIDER_RESPONSE_BODY_TEXT_RE.search(text):
        reasons.append(DIAGNOSTIC_VALIDATION_REASON_PROVIDER_RESPONSE_BODY)
    if _BROKER_RAW_MESSAGE_TEXT_RE.search(text):
        reasons.append(DIAGNOSTIC_VALIDATION_REASON_BROKER_RAW_MESSAGE)
    if _LOCAL_LLM_EXTRA_BODY_TEXT_RE.search(text):
        reasons.append(DIAGNOSTIC_VALIDATION_REASON_SENSITIVE_LOCAL_LLM_EXTRA_BODY)
    if _UNSAFE_TEXT_PAYLOAD_TEXT_RE.search(text):
        reasons.append(DIAGNOSTIC_VALIDATION_REASON_UNSAFE_TEXT_PAYLOAD)
    if not _is_safe_redaction_marker(text) and (
        _SENSITIVE_TOKEN_ASSIGNMENT_TEXT_RE.search(text)
        or _URL_USERINFO_RE.search(text)
        or any(pattern.search(text) for pattern in _SECRET_VALUE_PATTERNS)
    ):
        reasons.append(DIAGNOSTIC_VALIDATION_REASON_SECRET_PATTERN)
    if any(pattern.search(text) for pattern in _STACK_TRACE_PATTERNS):
        reasons.append(DIAGNOSTIC_VALIDATION_REASON_UNSAFE_TEXT_PAYLOAD)
    if _PRIVATE_KEY_BLOCK_TEXT_RE.search(text):
        reasons.append(DIAGNOSTIC_VALIDATION_REASON_SECRET_PATTERN)
    return tuple(dict.fromkeys(reasons))


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


def _redact_secret_assignment(match: re.Match[str]) -> str:
    return f"{match.group(1)}={DIAGNOSTIC_REDACTION_MARKER}"


def _copy_diagnostics_with_fields(
    diagnostics: ErrorDiagnostics,
    fields: Mapping[str, object],
) -> ErrorDiagnostics:
    return ErrorDiagnostics(
        component=diagnostics.component,
        operation=diagnostics.operation,
        code=diagnostics.code,
        category=diagnostics.category,
        visibility=diagnostics.visibility,
        content_policy=diagnostics.content_policy,
        status_code=diagnostics.status_code,
        retry_after_ms=diagnostics.retry_after_ms,
        fields=fields,
    )


def _is_supported_diagnostic_value(value: object) -> bool:
    return isinstance(value, str | int | float | bool) or value is None


def _max_depth(value: object) -> int:
    if isinstance(value, Mapping):
        if not value:
            return 1
        return 1 + max(_max_depth(child) for child in value.values())
    if isinstance(value, list | tuple | set | frozenset):
        if not value:
            return 1
        return 1 + max(_max_depth(child) for child in value)
    return 0


def _normalized_key(key: str) -> str:
    return _CAMEL_CASE_BOUNDARY_RE.sub("_", key).lower().replace("-", "_").replace(" ", "_")


def _key_segments(key: str) -> frozenset[str]:
    return frozenset(
        segment for segment in re.split(r"[.\[\]{}:/\\\s-]+", _normalized_key(key)) if segment
    )


def _is_sensitive_key(key: str) -> bool:
    normalized = _normalized_key(key)
    segments = _key_segments(normalized)
    if normalized in _SENSITIVE_TOKEN_KEYS or normalized.endswith("_token") or "token" in segments:
        return True
    return any(
        fragment != "token" and (fragment in normalized or fragment in segments)
        for fragment in _SENSITIVE_KEY_FRAGMENTS
    )


def _is_provider_response_body_key(key: str) -> bool:
    normalized = _normalized_key(key)
    compact = _compact_key(key)
    segments = _key_segments(normalized)
    return (
        normalized in _PROVIDER_RESPONSE_BODY_KEYS
        or compact in _PROVIDER_RESPONSE_BODY_KEYS
        or bool(_PROVIDER_RESPONSE_BODY_KEYS.intersection(segments))
    )


def _is_broker_raw_message_key(key: str) -> bool:
    normalized = _normalized_key(key)
    segments = _key_segments(normalized)
    if normalized in _BROKER_RAW_MESSAGE_KEYS:
        return True
    if "broker" not in segments and "broker" not in normalized:
        return False
    return (
        {"raw", "message"} <= segments
        or {"eligibility", "message"} <= segments
        or {"sensitive", "message"} <= segments
    )


def _is_local_llm_extra_body_key(key: str) -> bool:
    normalized = _normalized_key(key)
    return "extra_body" in normalized and (
        "local_llm" in normalized
        or "local_openai" in normalized
        or "local_llm" in _key_segments(key)
    )


def _is_unsafe_text_key(key: str) -> bool:
    normalized = _normalized_key(key)
    return normalized in _UNSAFE_TEXT_KEYS or bool(
        _UNSAFE_TEXT_KEYS.intersection(_key_segments(key))
    )


def _contains_unredacted_provider_response_body(key: str, value: object) -> bool:
    if _is_provider_response_body_key(key) and value != PROVIDER_RESPONSE_BODY_REDACTION_MARKER:
        return True
    return isinstance(value, str) and bool(_PROVIDER_RESPONSE_BODY_TEXT_RE.search(value))


def _contains_unredacted_broker_raw_message(key: str, value: object) -> bool:
    if _is_broker_raw_message_key(key) and value != BROKER_RAW_MESSAGE_REDACTION_MARKER:
        return True
    return isinstance(value, str) and bool(_BROKER_RAW_MESSAGE_TEXT_RE.search(value))


def _contains_unredacted_sensitive_local_llm_extra_body(key: str, value: object) -> bool:
    if value == LOCAL_LLM_EXTRA_BODY_REDACTION_MARKER:
        return False
    if isinstance(value, str) and _LOCAL_LLM_EXTRA_BODY_TEXT_RE.search(value):
        return True
    if not _is_local_llm_extra_body_key(key):
        return False
    if _LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS.intersection(_key_segments(key)):
        return True
    return _value_contains_sensitive_extra_body_key(value)


def _value_contains_sensitive_extra_body_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in _LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS
            or _value_contains_sensitive_extra_body_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list | tuple | set | frozenset):
        return any(_value_contains_sensitive_extra_body_key(child) for child in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(
            re.search(rf'"{re.escape(sensitive_key)}"\s*:', lowered)
            or re.search(rf"\b{re.escape(sensitive_key)}\s*[:=]", lowered)
            for sensitive_key in _LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS
        )
    return False


def _contains_unredacted_secret_pattern(key: str, value: object) -> bool:
    if _is_safe_redaction_marker(value):
        return False
    if _is_sensitive_key(key):
        return True
    return isinstance(value, str) and (
        bool(_SENSITIVE_TOKEN_ASSIGNMENT_TEXT_RE.search(value))
        or any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS)
    )


def _contains_unredacted_unsafe_text_payload(key: str, value: object) -> bool:
    if _is_safe_redaction_marker(value):
        return False
    if _is_unsafe_text_key(key):
        return True
    if isinstance(value, str) and _UNSAFE_TEXT_PAYLOAD_TEXT_RE.search(value):
        return True
    return isinstance(value, str) and any(
        pattern.search(value) for pattern in _STACK_TRACE_PATTERNS
    )


def _is_safe_redaction_marker(value: object) -> bool:
    return isinstance(value, str) and value in _SAFE_REDACTION_MARKERS


def _compact_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


__all__ = [
    "BROKER_RAW_MESSAGE_KEYS",
    "BROKER_RAW_MESSAGE_REDACTION_MARKER",
    "DEFAULT_DIAGNOSTIC_REDACTION_POLICY",
    "DESKTOP_RENDERER_EVENT_SCHEMA_VERSION",
    "DESKTOP_OVERLAY_REPRO_SCHEMA_VERSION",
    "DIAGNOSTIC_FIELD_MAX_DEPTH",
    "DIAGNOSTIC_REDACTION_MARKER",
    "DIAGNOSTIC_SINK_BASIC_LOGS",
    "DIAGNOSTIC_SINK_CHATBOX_DISCLOSURE",
    "DIAGNOSTIC_SINK_DASHBOARD",
    "DIAGNOSTIC_SINK_DETAILED_LOGS",
    "DIAGNOSTIC_SINK_FAILURE_JSONL",
    "DIAGNOSTIC_SINK_PERSISTED_LOGS",
    "DIAGNOSTIC_SINK_SNACKBAR",
    "DIAGNOSTIC_SINK_VISIBILITY_RULES",
    "DIAGNOSTIC_SINKS",
    "DIAGNOSTIC_VALIDATION_REASON_EXCESSIVE_DEPTH",
    "DIAGNOSTIC_VALIDATION_REASON_BROKER_RAW_MESSAGE",
    "DIAGNOSTIC_VALIDATION_REASON_FIELD_LIMIT_EXCEEDED",
    "DIAGNOSTIC_VALIDATION_REASON_PROVIDER_RESPONSE_BODY",
    "DIAGNOSTIC_VALIDATION_REASON_SECRET_PATTERN",
    "DIAGNOSTIC_VALIDATION_REASON_SENSITIVE_LOCAL_LLM_EXTRA_BODY",
    "DIAGNOSTIC_VALIDATION_REASON_UNSAFE_TEXT_PAYLOAD",
    "DIAGNOSTIC_VALIDATION_REASON_UNSUPPORTED_FIELD_TYPE",
    "DIAGNOSTIC_VALIDATION_REASON_VISIBILITY_FORBIDDEN",
    "DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED",
    "DIAGNOSTIC_VALIDATION_STATUS_REJECTED",
    "LOCAL_LLM_EXTRA_BODY_REDACTION_MARKER",
    "PROVIDER_RESPONSE_BODY_KEYS",
    "PROVIDER_RESPONSE_BODY_REDACTION_MARKER",
    "ContentPolicy",
    "DiagnosticRedactionPolicy",
    "DiagnosticSink",
    "DiagnosticTextValidationResult",
    "DiagnosticValidationReason",
    "DiagnosticValidationResult",
    "DiagnosticValidationStatus",
    "MESSAGE_PARAM_REDACTION_POLICY",
    "redact_diagnostics_for_sink",
    "redact_message_params_for_sink",
    "redact_text_for_sink",
    "redact_user_message_ref_for_sink",
    "validate_diagnostics_for_sink",
    "validate_desktop_renderer_event",
    "validate_desktop_overlay_repro_record",
    "validate_desktop_overlay_repro_result",
]
