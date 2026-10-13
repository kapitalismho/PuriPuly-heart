from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest

from puripuly_heart.core.diagnostic_validation import (
    DIAGNOSTIC_SINK_PERSISTED_LOGS,
    redact_diagnostics_for_sink,
)
from puripuly_heart.core.diagnostics import (
    DiagnosticHistory,
    DiagnosticQueryArgumentsError,
    current_diagnostic_operation,
    diagnostic_operation,
    normalize_diagnostic,
    safe_exception_fields,
)
from puripuly_heart.core.messages import ErrorDiagnostics


def diagnostic(**changes):
    values = dict(
        component="settings_repository",
        operation="save",
        code="settings_save_failed",
        category="transaction",
        visibility="basic",
        content_policy="metadata_only",
        status_code=None,
        retry_after_ms=None,
        fields={},
    )
    values.update(changes)
    return ErrorDiagnostics(**values)


def test_optional_rejection_keeps_cause_and_identity_without_private_content():
    original = diagnostic(
        fields={
            "message": "forbidden transcript sentinel",
            "errno": 13,
            "phase": "forbidden_sentinel",
            "exception_class": "PrivateSecretClass",
            "path": "C:/Users/private/settings.json",
        }
    )
    record = normalize_diagnostic(
        original,
        instance_id=str(uuid4()),
        severity="error",
        disposition="failed",
        stage="settings_save",
    )
    assert record["diagnostic_id"] == original.diagnostic_id
    assert record["code"] == "settings_save_failed"
    assert record["fields"] == {"errno": 13}
    assert record["detail"]["state"] == "rejected"
    assert record["detail"]["rejected_fields"] == 4
    text = json.dumps(record)
    assert "forbidden" not in text and "PrivateSecretClass" not in text and "C:/Users" not in text


def test_safe_exception_chain_retains_native_codes_not_text_or_private_class():
    native = PermissionError(13, "forbidden account token", "C:/Users/private/file")
    outer = ValueError("forbidden transcript")
    outer.__cause__ = native
    fields = safe_exception_fields(outer)
    assert fields == {
        "exception_class": "ValueError",
        "cause_class_1": "PermissionError",
        "errno": 13,
    }

    class PrivateSentinelError(Exception):
        pass

    assert (
        safe_exception_fields(PrivateSentinelError("forbidden"))["exception_class"]
        == "unclassified"
    )
    chain = [RuntimeError("forbidden") for _ in range(8)]
    for left, right in zip(chain, chain[1:]):
        left.__cause__ = right
    assert safe_exception_fields(chain[0])["cause_chain_truncated"] is True
    chain[-1].__cause__ = chain[0]
    assert len(safe_exception_fields(chain[0])) == 6


def test_redaction_projection_preserves_owner_record_and_cause_identity():
    original = diagnostic(
        cause_id=uuid4().hex, operation_id=uuid4().hex, turn_id=str(uuid4()), generation=3
    )
    result = redact_diagnostics_for_sink(original, DIAGNOSTIC_SINK_PERSISTED_LOGS)
    assert result.diagnostics is not None
    assert result.diagnostics.diagnostic_id == original.diagnostic_id
    assert result.diagnostics.cause_id == original.cause_id
    assert result.diagnostics.operation_id == original.operation_id
    assert result.diagnostics.turn_id == original.turn_id


def test_retention_dedup_filter_bounds_eviction_and_restart_are_explicit():
    history = DiagnosticHistory(max_records=3, max_bytes=4096)
    instance = str(uuid4())
    operation = uuid4().hex
    history.bind_instance(instance)
    causes = [diagnostic(operation_id=operation) for _ in range(5)]
    for cause in causes:
        history.retain(cause, severity="error", disposition="failed")
    replay, fresh = history.retain(causes[-1], severity="error", disposition="failed")
    assert fresh is False
    page = history.query({"operation_id": operation, "limit": 2})
    assert page["gap"] is True and page["truncated"] is True
    assert page["evicted_records"] == 2
    assert [record["diagnostic_id"] for record in page["records"]] == [
        cause.diagnostic_id for cause in causes[2:4]
    ]
    assert history.query({"diagnostic_id": causes[0].diagnostic_id})["lookup"] == "not_retained"
    assert history.query({"diagnostic_id": causes[-1].diagnostic_id})["records"] == [replay]
    assert history.query({"after": 99})["lookup"] == "invalid_cursor"
    other = DiagnosticHistory()
    other.bind_instance(str(uuid4()))
    prior = other.query({"instance_id": instance, "operation_id": operation})
    assert prior["lookup"] == "foreign_instance" and prior["gap"] is True
    assert prior["retention"]["restart_resets_history"] is True
    assert other.query({"operation_id": operation})["lookup"] == "not_retained"
    assert history.health()["history"]["duplicate_projections"] == 1
    for arguments in (
        {"limit": True},
        {"limit": 101},
        {"after": -1},
        {"diagnostic_id": "private prose"},
        {"path": "file"},
    ):
        with pytest.raises(ValueError):
            history.query(arguments)


@pytest.mark.parametrize("key", ("operation_id", "diagnostic_id"))
@pytest.mark.parametrize(
    "value",
    ("--------", "dead----", "overlay---------", "01234567-1234-1234-1234-123456789ab"),
)
def test_history_rejects_malformed_identity_filters(key, value):
    history = DiagnosticHistory()
    history.bind_instance("host_with-hyphen")
    with pytest.raises(DiagnosticQueryArgumentsError):
        history.query({key: value})


@pytest.mark.parametrize(
    "identity",
    (
        "deadbeef",
        "ABCDEF0123456789",
        "0123456789abcdef" * 2,
        "0123456789abcdef" * 4,
        "01234567-89ab-cdef-0123-456789abcdef",
        "overlay-0123456789abcdef",
        "overlay-01234567-89ab-cdef-0123-456789abcdef",
        "startup-" + "0123456789abcdef" * 2,
        "provider-01234567-89ab-cdef-0123-456789abcdef",
        "a_b-c-01234567",
    ),
)
def test_history_preserves_native_identity_formats(identity):
    history = DiagnosticHistory()
    history.bind_instance("host_with-hyphen")
    original = diagnostic(diagnostic_id=identity, operation_id=identity)
    record, fresh = history.retain(original, severity="error", disposition="failed")
    assert fresh
    assert record["diagnostic_id"] == identity
    assert record["operation_id"] == identity
    for key in ("diagnostic_id", "operation_id"):
        page = history.query({key: identity})
        assert page["lookup"] == "found"
        assert page["records"] == [record]
        assert history.query({key: "ffffffff"})["lookup"] == "not_retained"


@pytest.mark.asyncio
async def test_detached_scope_and_explicit_generations_cannot_inherit_stale_operation():
    instance = str(uuid4())
    operation = uuid4().hex
    request = uuid4().hex
    release = asyncio.Event()

    async def later():
        await release.wait()
        return normalize_diagnostic(
            diagnostic(), instance_id=instance, severity="error", disposition="failed"
        )

    with diagnostic_operation(instance, operation, request):
        captured = current_diagnostic_operation()
        assert captured == {"operation_id": operation, "request_id": request}
        current = normalize_diagnostic(
            diagnostic(), instance_id=instance, severity="error", disposition="failed"
        )
        assert current["operation_id"] == operation
        source = normalize_diagnostic(
            diagnostic(channel="self", generation=2),
            instance_id=instance,
            severity="error",
            disposition="failed",
        )
        assert source["operation_id"] is None
        explicit = normalize_diagnostic(
            diagnostic(channel="self", generation=2, **captured),
            instance_id=instance,
            severity="error",
            disposition="failed",
        )
        assert explicit["operation_id"] == operation
        detached = asyncio.create_task(later())
    release.set()
    assert (await detached)["operation_id"] is None
    assert current_diagnostic_operation() == {"operation_id": None, "request_id": None}


@pytest.mark.parametrize(
    "component,operation,code,stage,disposition",
    [
        (
            "output_runtime",
            "publish",
            "destination_publish_failed",
            "ui_event_delivery",
            "degraded",
        ),
        (
            "gpu_worker",
            "process_lifecycle",
            "worker_transport_closed_with_error",
            "writer_close",
            "degraded",
        ),
        (
            "settings_secrets",
            "create_store",
            "secret_store_passphrase_missing",
            "create_store",
            "failed",
        ),
        (
            "managed_connection_auth",
            "authorize_managed_connection",
            "managed_operation_authorization_expired",
            "remote_managed_operation_recovery",
            "failed",
        ),
        (
            "settings_application",
            "apply_ui_prompt_clipboard_state_runtime",
            "settings_memory_restore_failed",
            "restore_memory",
            "degraded",
        ),
    ],
)
def test_owned_finite_labels_survive_projection(component, operation, code, stage, disposition):
    record = normalize_diagnostic(
        diagnostic(component=component, operation=operation, code=code),
        instance_id=str(uuid4()),
        severity="warning",
        disposition=disposition,
        stage=stage,
    )
    assert (record["component"], record["operation"], record["code"], record["stage"]) == (
        component,
        operation,
        code,
        stage,
    )
    assert record["detail"]["rejected_fields"] == 0


@pytest.mark.parametrize(
    "code",
    [
        "window_reveal_lost",
        "window_observation_failed",
        "window_visibility_unstable",
        "window_bounds_failed",
        "window_native_ready_failed",
        "window_identity_failed",
    ],
)
def test_native_window_causes_survive_common_projection(code):
    record = normalize_diagnostic(
        diagnostic(
            component="overlay_process",
            operation="process_lifecycle",
            code=code,
            fields={"primary_failure_reason": code},
        ),
        instance_id=str(uuid4()),
        severity="error",
        disposition="failed",
        stage="native_terminal",
    )
    assert record["code"] == code
    assert record["fields"]["primary_failure_reason"] == code
    assert record["detail"]["rejected_fields"] == 0


def test_invalid_nonnull_top_level_details_are_counted_without_erasing_cause():
    source = diagnostic(
        fields={"errno": 13},
        cause_id="PRIVATE_CAUSE_MARKER",
        operation_id="PRIVATE_OPERATION_MARKER",
        request_id="PRIVATE_REQUEST_MARKER",
        turn_id="PRIVATE_TURN_MARKER",
        channel="PRIVATE_CHANNEL_MARKER",
        generation=-1,
        status_code=999,
        retry_after_ms=-1,
    )
    record = normalize_diagnostic(
        source, instance_id=str(uuid4()), severity="error", disposition="failed"
    )
    assert record["code"] == source.code and record["diagnostic_id"] == source.diagnostic_id
    assert all(
        record[key] is None
        for key in (
            "cause_id",
            "operation_id",
            "request_id",
            "turn_id",
            "channel",
            "generation",
            "status_code",
            "retry_after_ms",
        )
    )
    assert record["detail"]["state"] == "rejected"
    assert record["detail"]["rejected_fields"] == 8
    assert "PRIVATE_" not in json.dumps(record)
