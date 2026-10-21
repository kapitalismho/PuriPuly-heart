from __future__ import annotations

import json
import ssl
from dataclasses import replace
from uuid import uuid4

import httpx
import pytest
from websockets.datastructures import Headers
from websockets.exceptions import (
    ConnectionClosedError,
    ConnectionClosedOK,
    InvalidMessage,
    InvalidProxy,
    InvalidStatus,
    InvalidURI,
    PayloadTooBig,
    ProtocolError,
    ProxyError,
)
from websockets.frames import Close, CloseCode
from websockets.http11 import Response

from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
from puripuly_heart.core.diagnostic_validation import (
    validate_diagnostics_for_sink,
)
from puripuly_heart.core.diagnostics import normalize_diagnostic, sanitize_diagnostics
from puripuly_heart.core.error_messages import format_error_report_for_log, stt_failure_report
from puripuly_heart.core.stt.backend import STTProviderTurnIdentity
from puripuly_heart.core.stt.diagnostics import recognition_failure_diagnostics
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.providers.stt.qwen_audio import QwenAudioTaskFailedError
from puripuly_heart.providers.stt.qwen_audio_diagnostics import annotate_qwen_failure

PRIVATE_DETAIL = "token=qwen-private-detail-24963"


def annotate(failure: Exception, **kwargs) -> Exception:
    return annotate_qwen_failure(
        failure,
        channel=kwargs.pop("channel", "peer"),
        operation=kwargs.pop("operation", "readiness"),
        reason=kwargs.pop("reason", "provider_not_ready"),
        **kwargs,
    )


def retryable(failure: Exception) -> bool:
    engine = ScopedRecognitionEngine(session_factory=pytest.fail)
    return engine._retryable_connection_error(failure, "qwen_audio")


def identity() -> STTProviderTurnIdentity:
    return STTProviderTurnIdentity(
        AudioSegmentIdentity(3, 1, uuid4(), 4), uuid4().hex, uuid4().hex, ("qwen_audio",)
    )


@pytest.mark.parametrize(
    ("code", "category"),
    [
        (1000, "network"),
        (1001, "network"),
        (1006, "network"),
        (1011, "service_unavailable"),
        (1012, "service_unavailable"),
        (1013, "service_unavailable"),
    ],
)
def test_documented_temporary_websocket_closures_allow_pre_pcm_retry(code, category) -> None:
    failure = (
        ConnectionClosedOK(Close(code, PRIVATE_DETAIL), None)
        if code in {1000, 1001}
        else ConnectionClosedError(Close(code, PRIVATE_DETAIL), None)
    )
    annotated = annotate(failure, idle_ms=55000)

    assert annotated is failure
    assert annotated.readiness_retryable is True
    assert retryable(annotated) is True
    assert annotated.failure_diagnostics.category == category
    assert annotated.failure_diagnostics.fields["qwen_failure_kind"] == "websocket_close"
    assert annotated.failure_diagnostics.fields["ws_close_code"] == code
    assert annotated.failure_diagnostics.fields["idle_ms"] == 55000
    assert PRIVATE_DETAIL in str(annotated)
    assert PRIVATE_DETAIL not in repr(annotated.failure_diagnostics)


@pytest.mark.parametrize("sent", [None, Close(CloseCode.NORMAL_CLOSURE, PRIVATE_DETAIL)])
def test_no_received_close_frame_is_numeric_abnormal_closure(sent) -> None:
    failure = annotate(ConnectionClosedError(None, sent))

    assert failure.failure_diagnostics.fields["ws_close_code"] == 1006
    assert failure.readiness_retryable is True
    assert retryable(failure) is True


@pytest.mark.parametrize("code", [1002, 1003, 1005, 1007, 1008, 1009, 1010, 1014, 1015, 4001])
def test_protocol_policy_unsupported_and_unknown_websocket_closures_stop(code) -> None:
    failure = ConnectionClosedError(Close(code, "temporary timeout " + PRIVATE_DETAIL), None)
    failure.__cause__ = TimeoutError(PRIVATE_DETAIL)
    annotate(failure)

    assert failure.readiness_retryable is False
    assert retryable(failure) is False
    assert failure.failure_diagnostics.category == "invalid_response"
    assert failure.failure_diagnostics.fields["ws_close_code"] == code
    assert PRIVATE_DETAIL not in repr(failure.failure_diagnostics)


@pytest.mark.parametrize("received", [None, Close(CloseCode.NORMAL_CLOSURE, PRIVATE_DETAIL)])
def test_locally_sent_protocol_close_cannot_masquerade_as_retryable_eof(received) -> None:
    failure = annotate(
        ConnectionClosedError(
            received,
            Close(CloseCode.PROTOCOL_ERROR, PRIVATE_DETAIL),
            rcvd_then_sent=False if received is not None else None,
        )
    )

    assert failure.readiness_retryable is False
    assert retryable(failure) is False
    assert failure.failure_diagnostics.fields["ws_close_code"] == (
        1006 if received is None else 1000
    )


@pytest.mark.parametrize(
    "failure",
    [
        ProtocolError(PRIVATE_DETAIL),
        PayloadTooBig(1024, 512),
        InvalidMessage(PRIVATE_DETAIL),
        InvalidURI("wss://user:password@private.invalid", PRIVATE_DETAIL),
        InvalidProxy("http://user:password@private.invalid", PRIVATE_DETAIL),
        ProxyError(PRIVATE_DETAIL),
    ],
)
def test_typed_websocket_configuration_and_protocol_failures_do_not_use_transient_cause(
    failure,
) -> None:
    failure.__cause__ = ConnectionResetError(PRIVATE_DETAIL)
    annotate(failure)

    assert failure.readiness_retryable is False
    assert retryable(failure) is False
    assert PRIVATE_DETAIL not in repr(failure.failure_diagnostics)
    assert "private.invalid" not in repr(failure.failure_diagnostics)


@pytest.mark.parametrize(
    ("status", "category", "expected_retry"),
    [
        (400, "invalid_response", False),
        (401, "auth", False),
        (403, "auth", False),
        (404, "invalid_response", False),
        (408, "timeout", True),
        (429, "rate_limit", True),
        (500, "service_unavailable", True),
        (502, "service_unavailable", True),
        (503, "service_unavailable", True),
        (504, "service_unavailable", True),
    ],
)
def test_actual_websocket_handshake_status_uses_shared_http_classification(
    status, category, expected_retry
) -> None:
    failure = annotate(InvalidStatus(Response(status, PRIVATE_DETAIL, Headers(), b"private body")))

    assert failure.failure_diagnostics.category == category
    assert failure.failure_diagnostics.status_code == status
    assert failure.failure_diagnostics.fields["qwen_failure_kind"] == "websocket_handshake"
    assert failure.readiness_retryable is expected_retry
    assert retryable(failure) is expected_retry
    assert PRIVATE_DETAIL not in repr(failure.failure_diagnostics)
    assert "private body" not in repr(failure.failure_diagnostics)


@pytest.mark.parametrize(
    ("failure", "category", "expected_retry"),
    [
        (TimeoutError(PRIVATE_DETAIL), "timeout", True),
        (ConnectionResetError(PRIVATE_DETAIL), "network", True),
        (httpx.ConnectError(PRIVATE_DETAIL), "network", True),
        (httpx.ReadTimeout(PRIVATE_DETAIL), "timeout", True),
        (ssl.SSLCertVerificationError(1, PRIVATE_DETAIL), "network", False),
        (ssl.SSLError(1, PRIVATE_DETAIL), "network", False),
        (httpx.ProxyError(PRIVATE_DETAIL), "network", False),
        (FileNotFoundError(2, PRIVATE_DETAIL), "network", False),
        (PermissionError(13, PRIVATE_DETAIL), "network", False),
        (ValueError(PRIVATE_DETAIL), "unknown", False),
    ],
)
def test_ordinary_transport_failures_keep_shared_classification_and_safety_exclusions(
    failure, category, expected_retry
) -> None:
    annotate(failure)

    assert failure.failure_diagnostics.category == category
    assert not hasattr(failure, "readiness_retryable")
    assert retryable(failure) is expected_retry
    assert PRIVATE_DETAIL not in repr(failure.failure_diagnostics)


@pytest.mark.parametrize(
    "cause",
    [
        ssl.SSLCertVerificationError(1, PRIVATE_DETAIL),
        httpx.ProxyError(PRIVATE_DETAIL),
        FileNotFoundError(2, PRIVATE_DETAIL),
    ],
)
def test_temporary_websocket_close_cannot_override_transport_safety_exclusions(cause) -> None:
    failure = ConnectionClosedError(None, None)
    failure.__cause__ = cause
    annotate(failure)

    assert failure.readiness_retryable is False
    assert retryable(failure) is False


@pytest.mark.parametrize(
    ("code", "category", "expected_retry"),
    [
        ("CLIENT_ERROR", "invalid_response", False),
        ("InvalidApiKey", "auth", False),
        ("AccessDenied", "auth", False),
        ("AccessDenied.Unpurchased", "auth", False),
        ("InvalidParameter", "invalid_response", False),
        ("DataInspectionFailed", "invalid_response", False),
        ("ModelNotFound", "invalid_response", False),
        ("Throttling", "rate_limit", True),
        ("Throttling.RateQuota", "rate_limit", True),
        ("Throttling.BurstRate", "rate_limit", True),
        ("Throttling.AllocationQuota", "quota", False),
        ("Throttling.ServiceOverloaded", "service_unavailable", True),
        ("Throttling.ResourceExhausted", "service_unavailable", True),
        ("InternalError", "service_unavailable", True),
        ("InternalError.Timeout", "timeout", True),
        ("SystemError", "service_unavailable", True),
        ("RequestTimeOut", "timeout", True),
        ("ServiceUnavailable", "service_unavailable", True),
        ("ModelUnavailable", "service_unavailable", True),
    ],
)
def test_documented_native_codes_use_exact_conservative_allowlist(
    code, category, expected_retry
) -> None:
    failure = QwenAudioTaskFailedError(code, "request timeout after 23 seconds. " + PRIVATE_DETAIL)
    failure.__cause__ = TimeoutError(PRIVATE_DETAIL)
    annotated = annotate(failure, native_code=code)

    assert annotated is failure
    assert type(annotated) is QwenAudioTaskFailedError
    assert annotated.failure_diagnostics.category == category
    assert annotated.failure_diagnostics.fields["qwen_failure_kind"] == "native"
    assert annotated.failure_diagnostics.fields["qwen_native_code"] == code
    assert annotated.readiness_retryable is expected_retry
    assert retryable(annotated) is expected_retry
    assert PRIVATE_DETAIL not in repr(annotated.failure_diagnostics)
    assert annotated.error_code == code
    assert PRIVATE_DETAIL in annotated.error_message


@pytest.mark.parametrize(
    "code",
    [
        PRIVATE_DETAIL,
        "IDLE_TIMEOUT",
        "RequestTimeout",
        "InternalError.Algo.InvalidParameter",
        "CLIENT_ERROR.timeout",
        "503",
        "",
    ],
)
def test_unknown_native_codes_are_explicitly_nonretryable_and_redacted(code) -> None:
    failure = QwenAudioTaskFailedError(code, "timeout auth quota " + PRIVATE_DETAIL)
    failure.__cause__ = TimeoutError(PRIVATE_DETAIL)
    annotate(failure, native_code=code)

    assert failure.readiness_retryable is False
    assert retryable(failure) is False
    assert failure.failure_diagnostics.category == "unknown"
    assert failure.failure_diagnostics.fields["qwen_native_code"] == "unknown"
    assert PRIVATE_DETAIL not in repr(failure.failure_diagnostics)
    if len(code) > 3:
        assert code not in repr(failure.failure_diagnostics)


def test_client_error_timeout_prose_is_not_transient_authority() -> None:
    failure = annotate(
        QwenAudioTaskFailedError("CLIENT_ERROR", "request timeout after 23 seconds."),
        native_code="CLIENT_ERROR",
    )

    assert failure.failure_diagnostics.category == "invalid_response"
    assert failure.readiness_retryable is False
    assert retryable(failure) is False


def test_arbitrary_class_names_and_close_attributes_are_not_authority() -> None:
    failure_type = type("ConnectionClosedError_" + PRIVATE_DETAIL, (RuntimeError,), {})
    failure = failure_type("timeout " + PRIVATE_DETAIL)
    failure.code = 1013
    failure.rcvd = Close(1013, PRIVATE_DETAIL)
    annotate(failure)

    assert failure.failure_diagnostics.category == "unknown"
    assert failure.failure_diagnostics.fields["exception_class"] == "unclassified"
    assert "ws_close_code" not in failure.failure_diagnostics.fields
    assert not hasattr(failure, "readiness_retryable")
    assert retryable(failure) is False
    assert PRIVATE_DETAIL not in repr(failure.failure_diagnostics)


def test_reannotation_preserves_first_diagnostic_cause_and_source_identity() -> None:
    first_identity = identity()
    failure = annotate(
        QwenAudioTaskFailedError("InvalidApiKey", PRIVATE_DETAIL),
        native_code="InvalidApiKey",
        identity=first_identity,
        idle_ms=55000,
    )
    first = failure.failure_diagnostics
    annotate(
        failure,
        channel="self",
        operation="receive",
        reason="qwen_write_failed",
        native_code="ServiceUnavailable",
        identity=identity(),
        idle_ms=60000,
    )

    assert failure.failure_diagnostics == first
    assert failure.failure_diagnostics.diagnostic_id == first.diagnostic_id
    assert failure.failure_diagnostics.channel == "peer"
    assert failure.failure_diagnostics.operation == "readiness"
    assert failure.failure_diagnostics.turn_id == str(first_identity.segment.segment_id)
    assert failure.failure_diagnostics.fields["idle_ms"] == 55000
    assert failure.readiness_retryable is False


def test_annotated_native_root_cause_survives_cleanup_wrapper() -> None:
    original = annotate(
        QwenAudioTaskFailedError("InvalidApiKey", PRIVATE_DETAIL), native_code="InvalidApiKey"
    )
    wrapper = TimeoutError(PRIVATE_DETAIL)
    wrapper.__cause__ = original
    annotate(wrapper, operation="receive", reason="provider_cleanup_failed")

    assert wrapper.failure_diagnostics == original.failure_diagnostics
    assert wrapper.readiness_retryable is False
    assert retryable(wrapper) is False
    assert wrapper.__cause__ is original


@pytest.mark.parametrize("idle_ms", [None, True, -1, 2**64, 1.5, PRIVATE_DETAIL])
def test_unsafe_idle_measurements_are_omitted(idle_ms) -> None:
    failure = annotate(ConnectionClosedError(None, None), idle_ms=idle_ms)

    assert "idle_ms" not in failure.failure_diagnostics.fields
    assert (
        validate_diagnostics_for_sink(failure.failure_diagnostics, "persisted_logs").status
        == "accepted"
    )


def test_unlawful_operation_channel_and_identity_do_not_survive_annotation() -> None:
    unsafe_identity = STTProviderTurnIdentity(
        AudioSegmentIdentity(-1, 1, uuid4(), -1), PRIVATE_DETAIL, PRIVATE_DETAIL, ()
    )
    failure = annotate(
        RuntimeError(PRIVATE_DETAIL),
        channel=PRIVATE_DETAIL,
        operation=PRIVATE_DETAIL,
        reason=PRIVATE_DETAIL,
        identity=unsafe_identity,
    )
    diagnostics = failure.failure_diagnostics

    assert diagnostics.channel is None
    assert diagnostics.operation is None
    assert diagnostics.generation is None
    assert "provider_epoch_id" not in diagnostics.fields
    assert "provider_turn_id" not in diagnostics.fields
    assert "capture_epoch" not in diagnostics.fields
    assert diagnostics.fields["recognition_cause"] == "unclassified"
    assert PRIVATE_DETAIL not in repr(diagnostics)


def test_diagnostic_metadata_survives_shared_engine_and_persisted_projections_without_raw_detail() -> (
    None
):
    source_identity = identity()
    failure = annotate(
        ConnectionClosedError(Close(CloseCode.TRY_AGAIN_LATER, PRIVATE_DETAIL), None),
        identity=source_identity,
        idle_ms=55000,
    )
    attached = failure.failure_diagnostics
    report = stt_failure_report(failure, provider="qwen_audio", channel="peer", operation="receive")
    projected = recognition_failure_diagnostics(
        failure,
        provider="qwen_audio",
        channel="peer",
        operation="readiness",
        reason="provider_not_ready",
        identity=source_identity,
    )

    assert report.diagnostics == attached
    assert projected.diagnostic_id == attached.diagnostic_id
    assert projected.fields["ws_close_code"] == 1013
    assert projected.fields["qwen_failure_kind"] == "websocket_close"
    assert sanitize_diagnostics(projected) == projected
    for sink in (
        "dashboard",
        "snackbar",
        "chatbox_disclosure",
        "basic_logs",
        "persisted_logs",
        "failure_jsonl",
    ):
        diagnostics = replace(
            projected, visibility="persisted_failure_only" if sink == "failure_jsonl" else "basic"
        )
        validation = validate_diagnostics_for_sink(diagnostics, sink)
        assert validation.status == "accepted"
        assert validation.diagnostics.fields["idle_ms"] == 55000
    record = normalize_diagnostic(
        projected,
        instance_id=uuid4().hex,
        severity="error",
        disposition="failed",
        exception=failure,
    )
    rendered = json.dumps(record)
    assert record["fields"]["ws_close_code"] == 1013
    assert record["fields"]["idle_ms"] == 55000
    assert record["detail"]["rejected_fields"] == 0
    assert PRIVATE_DETAIL not in rendered
    assert PRIVATE_DETAIL not in format_error_report_for_log(report, sink="persisted_logs")
    assert "ConnectionClosedError" not in rendered


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("qwen_failure_kind", PRIVATE_DETAIL),
        ("qwen_native_code", PRIVATE_DETAIL),
        ("qwen_native_code", "InvalidParameter.arbitrary-suffix"),
        ("idle_ms", True),
        ("idle_ms", -1),
        ("idle_ms", 2**64),
        ("ws_close_code", True),
        ("ws_close_code", "1013"),
        ("ws_close_code", 5000),
    ],
)
def test_qwen_catalog_rejects_unbounded_values_before_and_after_sanitization(key, value) -> None:
    template = annotate(RuntimeError(PRIVATE_DETAIL)).failure_diagnostics
    diagnostics = replace(template, fields={key: value})

    assert validate_diagnostics_for_sink(diagnostics, "persisted_logs").status == "rejected"
    assert key not in sanitize_diagnostics(diagnostics).fields


@pytest.mark.parametrize(
    "cause",
    [
        ProtocolError(PRIVATE_DETAIL),
        InvalidStatus(Response(401, PRIVATE_DETAIL, Headers())),
        InvalidStatus(Response(403, PRIVATE_DETAIL, Headers())),
        InvalidStatus(Response(402, PRIVATE_DETAIL, Headers())),
        ConnectionClosedError(Close(CloseCode.POLICY_VIOLATION, PRIVATE_DETAIL), None),
    ],
)
def test_native_transient_code_cannot_override_typed_permanent_cause(cause) -> None:
    failure = QwenAudioTaskFailedError("ServiceUnavailable", PRIVATE_DETAIL)
    failure.__cause__ = cause
    annotate(failure, native_code="ServiceUnavailable")

    assert failure.readiness_retryable is False
    assert retryable(failure) is False


@pytest.mark.parametrize(
    "native_code", ["InvalidApiKey", "InvalidParameter", "CLIENT_ERROR", PRIVATE_DETAIL]
)
def test_native_permanent_code_cannot_fall_through_to_shared_service_failure(native_code) -> None:
    failure = QwenAudioTaskFailedError(native_code, PRIVATE_DETAIL)
    failure.__cause__ = InvalidStatus(Response(503, PRIVATE_DETAIL, Headers()))
    annotate(failure, native_code=native_code)

    assert failure.readiness_retryable is False
    assert retryable(failure) is False
    assert failure.failure_diagnostics.category not in {
        "service_unavailable",
        "rate_limit",
        "timeout",
    }
