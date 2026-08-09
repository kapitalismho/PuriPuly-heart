from __future__ import annotations

import importlib
import importlib.util

import pytest

from puripuly_heart.core import diagnostic_validation, error_messages, messages
from puripuly_heart.core.http_extensions import (
    HttpExtensionConfigurationError,
    HttpExtensionResponseError,
)
from puripuly_heart.core.messages import (
    DIAGNOSTIC_CATEGORY_AUTH,
    DIAGNOSTIC_CATEGORY_INVALID_RESPONSE,
    DIAGNOSTIC_CATEGORY_NETWORK,
    DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE,
    DIAGNOSTIC_CATEGORY_TIMEOUT,
    DiagnosticCategory,
)
from puripuly_heart.providers.extensions.http_extension_backend import (
    HttpExtensionTranslationError,
)

RAW_PROVIDER_DETAIL = "quota exceeded from upstream body token=provider-secret-123"
RAW_STT_DETAIL = "microphone socket closed bearer_token=stt-secret-456"


def _assert_raw_detail_absent(value: object, raw_detail: str) -> None:
    rendered = repr(value)
    assert raw_detail not in rendered
    assert "provider-secret-123" not in rendered
    assert "stt-secret-456" not in rendered


def test_provider_failure_report_maps_category_and_omits_raw_exception_text() -> None:
    spec = importlib.util.find_spec("puripuly_heart.core.error_messages")
    assert spec is not None, "provider/STT failure mapper module is missing"
    error_messages = importlib.import_module("puripuly_heart.core.error_messages")

    report = error_messages.provider_failure_report(
        RuntimeError(f"OpenRouter request failed (status=429, message={RAW_PROVIDER_DETAIL})"),
        provider="openrouter",
        operation="translate",
    )

    assert isinstance(report, messages.UserErrorReport)
    assert report.message.key == "provider.failure"
    assert report.message.params == {
        "category": messages.DIAGNOSTIC_CATEGORY_QUOTA,
        "operation": "translate",
        "provider": "openrouter",
    }
    assert report.diagnostics.category == messages.DIAGNOSTIC_CATEGORY_QUOTA
    assert report.diagnostics.code == "provider.quota"
    assert report.diagnostics.status_code == 429
    assert report.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY
    assert report.diagnostics.fields["exception_type"] == "RuntimeError"
    assert "raw_exception" not in report.diagnostics.fields
    assert "exception_text" not in report.diagnostics.fields
    _assert_raw_detail_absent(report, RAW_PROVIDER_DETAIL)
    _assert_raw_detail_absent(report.message.params, RAW_PROVIDER_DETAIL)
    _assert_raw_detail_absent(report.diagnostics.fields, RAW_PROVIDER_DETAIL)

    validation = diagnostic_validation.validate_diagnostics_for_sink(
        report.diagnostics,
        diagnostic_validation.DIAGNOSTIC_SINK_DASHBOARD,
    )
    assert validation.status == diagnostic_validation.DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED


@pytest.mark.parametrize(
    ("failure", "expected_category"),
    [
        (HttpExtensionTranslationError("connect error"), DIAGNOSTIC_CATEGORY_NETWORK),
        (HttpExtensionTranslationError("timeout"), DIAGNOSTIC_CATEGORY_TIMEOUT),
        (HttpExtensionResponseError("invalid response JSON"), DIAGNOSTIC_CATEGORY_INVALID_RESPONSE),
        (
            HttpExtensionConfigurationError(
                "missing required credential",
                diagnostic_category=DIAGNOSTIC_CATEGORY_AUTH,
            ),
            DIAGNOSTIC_CATEGORY_AUTH,
        ),
        (
            HttpExtensionTranslationError("HTTP status error", status_code=503),
            DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE,
        ),
    ],
)
def test_custom_http_failure_metadata_reaches_provider_diagnostics(
    failure: Exception,
    expected_category: DiagnosticCategory,
) -> None:
    wrapper = RuntimeError("translation request failed")
    wrapper.__cause__ = failure

    report = error_messages.provider_failure_report(
        wrapper,
        provider="llm",
        operation="translate",
    )

    assert report.message.params["provider"] == "custom_http"
    assert report.message.params["category"] == expected_category
    assert report.diagnostics.category == expected_category
    assert report.diagnostics.fields["provider"] == "custom_http"


def test_stt_failure_report_maps_network_category_and_keeps_diagnostics_metadata_only() -> None:
    spec = importlib.util.find_spec("puripuly_heart.core.error_messages")
    assert spec is not None, "provider/STT failure mapper module is missing"
    error_messages = importlib.import_module("puripuly_heart.core.error_messages")

    report = error_messages.stt_failure_report(
        ConnectionError(RAW_STT_DETAIL),
        provider="soniox",
        operation="open_session",
        channel="self",
        attempts=3,
    )

    assert isinstance(report, messages.UserErrorReport)
    assert report.message.key == "stt.failure"
    assert report.message.params == {
        "category": messages.DIAGNOSTIC_CATEGORY_NETWORK,
        "operation": "open_session",
        "provider": "soniox",
    }
    assert report.diagnostics.category == messages.DIAGNOSTIC_CATEGORY_NETWORK
    assert report.diagnostics.code == "stt.network"
    assert report.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY
    assert report.diagnostics.fields == {
        "attempts": 3,
        "channel": "self",
        "exception_type": "ConnectionError",
        "provider": "soniox",
    }
    _assert_raw_detail_absent(report, RAW_STT_DETAIL)
    _assert_raw_detail_absent(report.message.params, RAW_STT_DETAIL)
    _assert_raw_detail_absent(report.diagnostics.fields, RAW_STT_DETAIL)

    validation = diagnostic_validation.validate_diagnostics_for_sink(
        report.diagnostics,
        diagnostic_validation.DIAGNOSTIC_SINK_BASIC_LOGS,
    )
    assert validation.status == diagnostic_validation.DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED


def test_format_error_report_for_log_falls_back_when_diagnostics_are_unsafe() -> None:
    error_messages = importlib.import_module("puripuly_heart.core.error_messages")
    raw_detail = "upstream payload failed token=formatter-secret-123"
    report = messages.UserErrorReport(
        message=messages.UserMessageRef(
            key="provider.failure",
            params={"category": messages.DIAGNOSTIC_CATEGORY_UNKNOWN},
            severity=messages.SEVERITY_ERROR,
        ),
        diagnostics=messages.ErrorDiagnostics(
            component="provider.llm",
            operation="translate",
            code="provider.unknown",
            category=messages.DIAGNOSTIC_CATEGORY_UNKNOWN,
            visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
            content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
            status_code=None,
            retry_after_ms=None,
            fields={
                "managed_code": raw_detail,
                "raw_exception": raw_detail,
            },
        ),
    )

    rendered = error_messages.format_error_report_for_log(report)

    assert rendered == "category=unknown code=diagnostics.rejected diagnostic_status=rejected"
    assert raw_detail not in rendered
    assert "formatter-secret-123" not in rendered
    assert "managed_code" not in rendered
    assert "raw_exception" not in rendered
    assert "secret_pattern" not in rendered
