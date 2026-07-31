from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from typing import Any, get_args

import pytest

from puripuly_heart.core import messages
from tests.helpers.ast_sources import imported_modules

FORBIDDEN_IMPORT_PREFIXES = (
    "flet",
    "puripuly_heart.app.adapters",
    "puripuly_heart.config.settings",
    "puripuly_heart.core.managed_openrouter_broker_client",
    "puripuly_heart.core.osc",
    "puripuly_heart.core.runtime_logging",
    "puripuly_heart.providers",
    "puripuly_heart.ui",
)


def _validator():
    return importlib.import_module("puripuly_heart.core.diagnostic_validation")


def _forbidden_imports(module: object) -> set[str]:
    imports = imported_modules(Path(getattr(module, "__file__") or ""))
    return {
        imported
        for imported in imports
        for forbidden in FORBIDDEN_IMPORT_PREFIXES
        if imported == forbidden or imported.startswith(f"{forbidden}.")
    }


def _diagnostics(
    *,
    fields: dict[str, Any] | None = None,
    visibility: messages.DiagnosticVisibility = messages.DIAGNOSTIC_VISIBILITY_DETAILED,
    content_policy: messages.ContentPolicy = messages.CONTENT_POLICY_METADATA_ONLY,
) -> messages.ErrorDiagnostics:
    return messages.ErrorDiagnostics(
        component="runtime.apply",
        operation="provider_call",
        code="provider_failure",
        category=messages.DIAGNOSTIC_CATEGORY_INVALID_RESPONSE,
        visibility=visibility,
        content_policy=content_policy,
        status_code=502,
        retry_after_ms=None,
        fields=fields or {"provider": "openrouter", "attempt": 1},
    )


def test_diagnostic_sink_contract_covers_live_logs_persisted_and_failure_jsonl() -> None:
    validator = _validator()

    expected_sinks = {
        validator.DIAGNOSTIC_SINK_DASHBOARD,
        validator.DIAGNOSTIC_SINK_SNACKBAR,
        validator.DIAGNOSTIC_SINK_CHATBOX_DISCLOSURE,
        validator.DIAGNOSTIC_SINK_BASIC_LOGS,
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
        validator.DIAGNOSTIC_SINK_PERSISTED_LOGS,
        validator.DIAGNOSTIC_SINK_FAILURE_JSONL,
    }
    assert set(get_args(validator.DiagnosticSink)) == expected_sinks
    assert set(validator.DIAGNOSTIC_SINKS) == expected_sinks
    assert set(validator.DIAGNOSTIC_SINK_VISIBILITY_RULES) == expected_sinks

    compatible_visibility_by_sink = {
        validator.DIAGNOSTIC_SINK_DASHBOARD: messages.DIAGNOSTIC_VISIBILITY_BASIC,
        validator.DIAGNOSTIC_SINK_SNACKBAR: messages.DIAGNOSTIC_VISIBILITY_BASIC,
        validator.DIAGNOSTIC_SINK_CHATBOX_DISCLOSURE: messages.DIAGNOSTIC_VISIBILITY_BASIC,
        validator.DIAGNOSTIC_SINK_BASIC_LOGS: messages.DIAGNOSTIC_VISIBILITY_BASIC,
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS: messages.DIAGNOSTIC_VISIBILITY_DETAILED,
        validator.DIAGNOSTIC_SINK_PERSISTED_LOGS: messages.DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
        validator.DIAGNOSTIC_SINK_FAILURE_JSONL: messages.DIAGNOSTIC_VISIBILITY_PERSISTED_FAILURE_ONLY,
    }

    for sink, visibility in compatible_visibility_by_sink.items():
        result = validator.validate_diagnostics_for_sink(
            _diagnostics(visibility=visibility),
            sink,
        )
        assert result.status == validator.DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED
        assert result.diagnostics is not None
        assert result.diagnostics.fields["provider"] == "openrouter"

    result = validator.validate_diagnostics_for_sink(
        _diagnostics(visibility=messages.DIAGNOSTIC_VISIBILITY_PERSISTED_FAILURE_ONLY),
        validator.DIAGNOSTIC_SINK_DASHBOARD,
    )
    assert result.status == validator.DIAGNOSTIC_VALIDATION_STATUS_REJECTED
    assert validator.DIAGNOSTIC_VALIDATION_REASON_VISIBILITY_FORBIDDEN in result.reasons


def test_validator_rejects_unsupported_field_types_before_sink_output() -> None:
    validator = _validator()

    result = validator.validate_diagnostics_for_sink(
        _diagnostics(fields={"provider": "openrouter", "raw_headers": ["x-request-id"]}),
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
    )

    assert result.status == validator.DIAGNOSTIC_VALIDATION_STATUS_REJECTED
    assert validator.DIAGNOSTIC_VALIDATION_REASON_UNSUPPORTED_FIELD_TYPE in result.reasons
    assert result.diagnostics is None


def test_validator_rejects_excessive_field_depth_and_size() -> None:
    validator = _validator()

    too_many_fields = {
        f"field_{index}": index for index in range(messages.DIAGNOSTIC_FIELD_MAX_ITEMS + 1)
    }
    size_result = validator.validate_diagnostics_for_sink(
        _diagnostics(fields=too_many_fields),
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
    )
    assert size_result.status == validator.DIAGNOSTIC_VALIDATION_STATUS_REJECTED
    assert validator.DIAGNOSTIC_VALIDATION_REASON_FIELD_LIMIT_EXCEEDED in size_result.reasons

    too_deep: object = "leaf"
    for index in range(validator.DIAGNOSTIC_FIELD_MAX_DEPTH + 1):
        too_deep = {f"level_{index}": too_deep}
    depth_result = validator.validate_diagnostics_for_sink(
        _diagnostics(fields={"nested": too_deep}),
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
    )
    assert depth_result.status == validator.DIAGNOSTIC_VALIDATION_STATUS_REJECTED
    assert validator.DIAGNOSTIC_VALIDATION_REASON_EXCESSIVE_DEPTH in depth_result.reasons


def test_validator_rejects_known_secret_patterns_and_redactor_removes_them() -> None:
    validator = _validator()

    unsafe = _diagnostics(
        content_policy=messages.CONTENT_POLICY_REDACTED,
        fields={"provider": "openrouter", "error": "Authorization: Bearer sk-live-secret-token"},
    )

    validation = validator.validate_diagnostics_for_sink(
        unsafe,
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
    )
    assert validation.status == validator.DIAGNOSTIC_VALIDATION_STATUS_REJECTED
    assert validator.DIAGNOSTIC_VALIDATION_REASON_SECRET_PATTERN in validation.reasons

    redacted = validator.redact_diagnostics_for_sink(
        unsafe,
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
    )
    assert redacted.status == validator.DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED
    assert redacted.redacted is True
    assert redacted.diagnostics is not None
    rendered = redacted.diagnostics.fields["error"]
    assert rendered == validator.DIAGNOSTIC_REDACTION_MARKER
    assert "sk-live-secret-token" not in rendered


def test_validator_rejects_token_assignment_variants_inside_values() -> None:
    validator = _validator()
    unsafe = _diagnostics(
        content_policy=messages.CONTENT_POLICY_REDACTED,
        fields={
            "provider": "openrouter",
            "error": (
                "request failed access_token=structured-access-secret "
                "refreshToken=structured-refresh-secret "
                "idToken=structured-id-secret"
            ),
        },
    )

    validation = validator.validate_diagnostics_for_sink(
        unsafe,
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
    )
    assert validation.status == validator.DIAGNOSTIC_VALIDATION_STATUS_REJECTED
    assert validator.DIAGNOSTIC_VALIDATION_REASON_SECRET_PATTERN in validation.reasons

    redacted = validator.redact_diagnostics_for_sink(
        unsafe,
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
    )
    assert redacted.status == validator.DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED
    assert redacted.redacted is True
    assert redacted.diagnostics is not None
    rendered = repr(redacted.diagnostics.fields)
    assert "structured-access-secret" not in rendered
    assert "structured-refresh-secret" not in rendered
    assert "structured-id-secret" not in rendered


def test_text_redactor_removes_token_assignment_key_variants() -> None:
    validator = _validator()
    text = (
        "provider failed access_token=text-access-secret "
        "refreshToken=text-refresh-secret "
        "idToken=text-id-secret"
    )

    result = validator.redact_text_for_sink(text, validator.DIAGNOSTIC_SINK_DETAILED_LOGS)

    assert result.status == validator.DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED
    assert result.redacted is True
    assert result.text is not None
    assert "text-access-secret" not in result.text
    assert "text-refresh-secret" not in result.text
    assert "text-id-secret" not in result.text
    assert validator.DIAGNOSTIC_REDACTION_MARKER in result.text


@pytest.mark.parametrize(
    "payload",
    [
        '{"password":"hunter2"}',
        '{"api_key":"sk-json-secret"}',
        '{"authorization":"Bearer json-secret-token"}',
        '{"token":"json-secret-token"}',
    ],
)
def test_validator_rejects_quoted_json_secret_labels_inside_values(payload: str) -> None:
    validator = _validator()
    unsafe = _diagnostics(
        content_policy=messages.CONTENT_POLICY_REDACTED,
        fields={"provider": "openrouter", "error": payload},
    )

    validation = validator.validate_diagnostics_for_sink(
        unsafe,
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
    )
    assert validation.status == validator.DIAGNOSTIC_VALIDATION_STATUS_REJECTED
    assert validator.DIAGNOSTIC_VALIDATION_REASON_SECRET_PATTERN in validation.reasons

    redacted = validator.redact_diagnostics_for_sink(
        unsafe,
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
    )
    assert redacted.status == validator.DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED
    assert redacted.redacted is True
    assert redacted.diagnostics is not None
    assert redacted.diagnostics.fields["error"] == validator.DIAGNOSTIC_REDACTION_MARKER


def test_token_metric_field_names_do_not_weaken_auth_token_rejection() -> None:
    validator = _validator()

    safe_metrics = validator.validate_diagnostics_for_sink(
        _diagnostics(fields={"prompt_tokens": 12, "completion_tokens": 8, "token_count": 20}),
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
    )
    assert safe_metrics.status == validator.DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED

    for sensitive_key in (
        "token",
        "session_token",
        "access_token",
        "bearer_token",
        "accessToken",
        "refreshToken",
        "idToken",
        "authToken",
    ):
        rejected = validator.validate_diagnostics_for_sink(
            _diagnostics(fields={sensitive_key: "do-not-log"}),
            validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
        )
        assert rejected.status == validator.DIAGNOSTIC_VALIDATION_STATUS_REJECTED
        assert validator.DIAGNOSTIC_VALIDATION_REASON_SECRET_PATTERN in rejected.reasons


def test_validator_rejects_unredacted_provider_response_bodies() -> None:
    validator = _validator()

    result = validator.validate_diagnostics_for_sink(
        _diagnostics(
            fields={
                "provider": "openrouter",
                "provider_response_body": '{"error":{"message":"invalid key"}}',
            },
        ),
        validator.DIAGNOSTIC_SINK_FAILURE_JSONL,
    )

    assert result.status == validator.DIAGNOSTIC_VALIDATION_STATUS_REJECTED
    assert validator.DIAGNOSTIC_VALIDATION_REASON_PROVIDER_RESPONSE_BODY in result.reasons


@pytest.mark.parametrize(
    "field_name",
    ["provider.response.body", "raw.response.body", "provider-response-body", "raw-response-body"],
)
def test_validator_rejects_provider_response_body_path_style_keys(field_name: str) -> None:
    validator = _validator()
    unsafe = _diagnostics(
        visibility=messages.DIAGNOSTIC_VISIBILITY_PERSISTED_FAILURE_ONLY,
        content_policy=messages.CONTENT_POLICY_REDACTED,
        fields={field_name: '{"error":{"message":"raw provider payload"}}'},
    )

    validation = validator.validate_diagnostics_for_sink(
        unsafe,
        validator.DIAGNOSTIC_SINK_FAILURE_JSONL,
    )
    assert validation.status == validator.DIAGNOSTIC_VALIDATION_STATUS_REJECTED
    assert validator.DIAGNOSTIC_VALIDATION_REASON_PROVIDER_RESPONSE_BODY in validation.reasons

    redacted = validator.redact_diagnostics_for_sink(
        unsafe,
        validator.DIAGNOSTIC_SINK_FAILURE_JSONL,
    )
    assert redacted.status == validator.DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED
    assert redacted.redacted is True
    assert redacted.diagnostics is not None
    assert (
        redacted.diagnostics.fields[field_name] == validator.PROVIDER_RESPONSE_BODY_REDACTION_MARKER
    )


def test_validator_rejects_and_redacts_raw_broker_messages() -> None:
    validator = _validator()
    unsafe = _diagnostics(
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_REDACTED,
        fields={
            "provider": "managed_openrouter",
            "broker_eligibility_message": "eligibility check returned raw broker details",
        },
    )

    validation = validator.validate_diagnostics_for_sink(
        unsafe,
        validator.DIAGNOSTIC_SINK_BASIC_LOGS,
    )
    assert validation.status == validator.DIAGNOSTIC_VALIDATION_STATUS_REJECTED
    assert validator.DIAGNOSTIC_VALIDATION_REASON_BROKER_RAW_MESSAGE in validation.reasons

    redacted = validator.redact_diagnostics_for_sink(
        unsafe,
        validator.DIAGNOSTIC_SINK_BASIC_LOGS,
    )
    assert redacted.status == validator.DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED
    assert redacted.redacted is True
    assert redacted.diagnostics is not None
    assert (
        redacted.diagnostics.fields["broker_eligibility_message"]
        == validator.BROKER_RAW_MESSAGE_REDACTION_MARKER
    )


def test_validator_rejects_managed_private_key_material_inside_values() -> None:
    validator = _validator()
    unsafe = _diagnostics(
        fields={
            "provider": "managed_openrouter",
            "broker_error": '{"managed_private_key":"do-not-log-this"}',
            "alternate_error": "managed-private-key=do-not-log-this-either",
        },
    )

    validation = validator.validate_diagnostics_for_sink(
        unsafe,
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
    )
    assert validation.status == validator.DIAGNOSTIC_VALIDATION_STATUS_REJECTED
    assert validator.DIAGNOSTIC_VALIDATION_REASON_SECRET_PATTERN in validation.reasons

    redacted = validator.redact_diagnostics_for_sink(
        _diagnostics(
            content_policy=messages.CONTENT_POLICY_REDACTED,
            fields={"broker_error": '{"managed_private_key":"do-not-log-this"}'},
        ),
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
    )
    assert redacted.status == validator.DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED
    assert redacted.redacted is True
    assert redacted.diagnostics is not None
    assert redacted.diagnostics.fields["broker_error"] == validator.DIAGNOSTIC_REDACTION_MARKER


@pytest.mark.parametrize(
    "field_name",
    [
        "transcript_text",
        "translation_text",
        "source_text",
        "raw_transcript",
        "raw_translation",
        "raw_source_text",
        "translated_text",
        "payload.transcript-text",
        "payload.translationText",
        "payload.sourceText",
    ],
)
def test_validator_rejects_and_redacts_raw_transcript_translation_source_fields(
    field_name: str,
) -> None:
    validator = _validator()
    unsafe = _diagnostics(
        visibility=messages.DIAGNOSTIC_VISIBILITY_DETAILED,
        content_policy=messages.CONTENT_POLICY_REDACTED,
        fields={field_name: "raw user utterance text must not enter diagnostics"},
    )

    validation = validator.validate_diagnostics_for_sink(
        unsafe,
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
    )
    assert validation.status == validator.DIAGNOSTIC_VALIDATION_STATUS_REJECTED
    assert validator.DIAGNOSTIC_VALIDATION_REASON_UNSAFE_TEXT_PAYLOAD in validation.reasons

    redacted = validator.redact_diagnostics_for_sink(
        unsafe,
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
    )
    assert redacted.status == validator.DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED
    assert redacted.redacted is True
    assert redacted.diagnostics is not None
    assert redacted.diagnostics.fields[field_name] == validator.DIAGNOSTIC_REDACTION_MARKER


def test_text_redactor_removes_raw_transcript_translation_source_assignments() -> None:
    validator = _validator()
    text = (
        "transcript_text=hello secret transcript\n"
        "translationText=bonjour secret translation\n"
        "source text=original source text"
    )

    result = validator.redact_text_for_sink(text, validator.DIAGNOSTIC_SINK_DETAILED_LOGS)

    assert result.status == validator.DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED
    assert result.redacted is True
    assert result.text is not None
    assert "hello secret transcript" not in result.text
    assert "bonjour secret translation" not in result.text
    assert "original source text" not in result.text
    assert validator.DIAGNOSTIC_REDACTION_MARKER in result.text


@pytest.mark.parametrize(
    ("text", "forbidden", "marker"),
    [
        (
            'provider_response_body={"error":{"message":"raw provider payload"}}',
            "raw provider payload",
            "PROVIDER_RESPONSE_BODY_REDACTION_MARKER",
        ),
        (
            "broker raw message=raw broker credential assertion failure",
            "raw broker credential assertion failure",
            "BROKER_RAW_MESSAGE_REDACTION_MARKER",
        ),
        (
            'raw_exception=Traceback (most recent call last): File "provider.py", line 1',
            "provider.py",
            "DIAGNOSTIC_REDACTION_MARKER",
        ),
        (
            "Authorization: Bearer provider-secret-token",
            "provider-secret-token",
            "DIAGNOSTIC_REDACTION_MARKER",
        ),
    ],
)
def test_text_redactor_removes_provider_broker_exception_and_secret_log_payloads(
    text: str,
    forbidden: str,
    marker: str,
) -> None:
    validator = _validator()

    result = validator.redact_text_for_sink(text, validator.DIAGNOSTIC_SINK_PERSISTED_LOGS)

    assert result.redacted is True
    if result.status == validator.DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED:
        assert result.text is not None
        assert forbidden not in result.text
        assert getattr(validator, marker) in result.text
    else:
        assert result.status == validator.DIAGNOSTIC_VALIDATION_STATUS_REJECTED
        assert result.text is None


def test_local_llm_sensitive_extra_body_requires_named_allow_policy() -> None:
    validator = _validator()
    unsafe = _diagnostics(
        visibility=messages.DIAGNOSTIC_VISIBILITY_PERSISTED_FAILURE_ONLY,
        content_policy=messages.CONTENT_POLICY_REDACTED,
        fields={"local_llm.extra_body": '{"authorization":"Bearer local-secret"}'},
    )

    default_result = validator.redact_diagnostics_for_sink(
        unsafe,
        validator.DIAGNOSTIC_SINK_FAILURE_JSONL,
    )
    assert default_result.status == validator.DIAGNOSTIC_VALIDATION_STATUS_REJECTED
    assert (
        validator.DIAGNOSTIC_VALIDATION_REASON_SENSITIVE_LOCAL_LLM_EXTRA_BODY
        in default_result.reasons
    )

    allow_policy = validator.DiagnosticRedactionPolicy(
        allow_sensitive_local_llm_extra_body_fields=True,
    )
    allowed = validator.redact_diagnostics_for_sink(
        unsafe,
        validator.DIAGNOSTIC_SINK_FAILURE_JSONL,
        policy=allow_policy,
    )

    assert allowed.status == validator.DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED
    assert allowed.redacted is True
    assert allowed.diagnostics is not None
    assert (
        allowed.diagnostics.fields["local_llm.extra_body"]
        == validator.LOCAL_LLM_EXTRA_BODY_REDACTION_MARKER
    )


def test_validator_contract_results_are_immutable_and_import_safe() -> None:
    validator = _validator()

    assert not _forbidden_imports(validator)

    policy = validator.DiagnosticRedactionPolicy()
    result = validator.validate_diagnostics_for_sink(
        _diagnostics(),
        validator.DIAGNOSTIC_SINK_DETAILED_LOGS,
        policy=policy,
    )

    assert is_dataclass(policy)
    assert is_dataclass(result)
    assert not hasattr(policy, "__dict__")
    assert not hasattr(result, "__dict__")
    assert isinstance(result.reasons, tuple)
    with pytest.raises(FrozenInstanceError):
        policy.allow_sensitive_local_llm_extra_body_fields = True  # type: ignore[misc]
    assert result.diagnostics is not None
    with pytest.raises(TypeError):
        result.diagnostics.fields["provider"] = "qwen"  # type: ignore[index]
