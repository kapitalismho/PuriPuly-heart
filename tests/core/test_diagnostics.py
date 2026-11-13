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
    emit_failure,
    exception_diagnostics,
    normalize_diagnostic,
    project_diagnostic_record,
    safe_exception_fields,
    sanitize_diagnostic_fields,
    sanitize_diagnostics,
    serialize_diagnostics,
)
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.network_diagnostics import TRANSPORT_ENUM_FIELDS


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


def test_caught_up_cursor_distinguishes_new_records_from_missing_identity():
    history = DiagnosticHistory(max_records=1)
    history.bind_instance(str(uuid4()))
    first, _ = history.retain(diagnostic(), severity="error", disposition="failed")
    cursor = first["diagnostic_sequence"]
    caught_up = history.query({"after": cursor})
    assert caught_up["lookup"] == "no_new_records"
    assert caught_up["records"] == []
    assert caught_up["next_after"] == caught_up["latest_sequence"] == cursor
    assert caught_up["gap"] is False
    assert (
        history.query(
            {
                "after": cursor,
                "diagnostic_id": uuid4().hex,
            }
        )["lookup"]
        == "not_retained"
    )
    second, _ = history.retain(diagnostic(), severity="error", disposition="failed")
    resumed = history.query({"after": cursor})
    assert resumed["lookup"] == "found"
    assert resumed["records"] == [second]
    assert resumed["gap"] is False
    assert history.query({"after": 0})["gap"] is True


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
            "asr_worker",
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


def test_audio_diagnosis_retains_only_actual_bounded_numeric_measurements():
    fields = {
        "target_index": 7,
        "opened": True,
        "frames_received": True,
        "frame_count": 3,
        "vad_sample_count": 12,
        "speech_probability_min": 0.1,
        "speech_probability_max": 0.72,
        "speech_threshold": 0.95,
        "speech_probability_ppm": 720000,
        "level_ppm": 400000,
        "speech_detected": False,
        "silence_observed": False,
        "cleanup_complete": True,
        "audio_failure_code": "none",
        "audio_restore_self": "restored",
        "audio_restore_peer": "not_needed",
    }
    record = normalize_diagnostic(
        diagnostic(
            component="audio_diagnosis",
            operation="audio.diagnose",
            code="audio_diagnose_candidate",
            fields=fields,
        ),
        instance_id=str(uuid4()),
        severity="info",
        disposition="recovered",
        stage="audio_diagnosis",
    )
    assert record["component"] == "audio_diagnosis"
    assert record["operation"] == "audio.diagnose"
    assert record["fields"] == fields
    assert record["detail"]["state"] == "available"


@pytest.mark.parametrize("value", (-0.1, 1.1, float("nan"), float("inf"), True, "0.7", 10**400))
def test_audio_diagnostic_probability_fields_reject_non_measurements(value):
    fields, rejected = sanitize_diagnostic_fields(
        {
            "speech_probability_min": value,
            "speech_probability_max": value,
            "speech_threshold": value,
        }
    )
    assert fields == {} and rejected == 3


def test_audio_diagnostic_history_preserves_scalar_metadata_boundary():
    fields, rejected = sanitize_diagnostic_fields(
        {
            "observed_speech_probabilities": [0.1, 0.55, 0.72],
            "background_probability_max": 0.1,
            "synthetic": True,
            "settings_revision": 9,
            "speech_probability_ppm": 1000001,
            "level_ppm": -1,
        }
    )
    assert fields == {} and rejected == 6


def test_common_failure_views_preserve_safe_nested_evidence_and_identity():
    cause = diagnostic(
        status_code=403,
        fields={"errno": 13, "message": "PRIVATE_SECRET C:/Users/private/file"},
    )
    original = diagnostic(causes=(cause,), fields={"count": 2})
    value = serialize_diagnostics(original)
    history = DiagnosticHistory()
    history.bind_instance("test-instance")
    retained, _ = history.retain(original, severity="warning", disposition="recovered")
    projected = project_diagnostic_record(
        {**retained, "raw_body": "PRIVATE_SECRET", "file_evidence": ["PRIVATE_SECRET"]}
    )

    assert value["diagnostic_id"] == original.diagnostic_id
    assert value["causes"][0]["diagnostic_id"] == cause.diagnostic_id
    assert value["causes"][0]["fields"] == {"errno": 13}
    assert value["causes"][0]["rejected_fields"] == 1
    assert projected["causes"] == value["causes"]
    assert projected["diagnostic_sequence"] == retained["diagnostic_sequence"]
    assert projected["disposition"] == "recovered"
    assert projected["fields"] == {"count": 2}
    assert projected["file_evidence"] == "unknown"
    assert "PRIVATE_SECRET" not in json.dumps(projected)
    assert "PRIVATE_SECRET" not in json.dumps(value)
    assert serialize_diagnostics(sanitize_diagnostics(original)) == value


def test_failure_tree_overflow_keeps_first_meaningful_causes_with_explicit_loss():
    children = tuple(diagnostic(status_code=403 if index == 0 else 404) for index in range(20))
    original = diagnostic(causes=children)
    record = normalize_diagnostic(
        original, instance_id=None, severity="error", disposition="failed"
    )

    assert 0 < len(record["causes"]) < len(children)
    assert record["causes"][0]["diagnostic_id"] == children[0].diagnostic_id
    assert record["causes"][0]["status_code"] == 403
    assert record["causes_truncated"] is True
    assert record["detail"]["state"] == "truncated"
    assert len(json.dumps(record, separators=(",", ":")).encode()) <= 4096
    assert len(json.dumps(serialize_diagnostics(original), separators=(",", ":")).encode()) <= 4096


def test_failure_projector_bounds_recursive_external_evidence():
    raw = serialize_diagnostics(diagnostic())
    raw["causes"] = [raw]
    raw["fields"]["path"] = "C:/Users/PRIVATE_SECRET"
    record = project_diagnostic_record(raw)

    assert record["diagnostic_id"] == raw["diagnostic_id"]
    assert record["detail"]["truncated"] is True
    assert "PRIVATE_SECRET" not in json.dumps(record)
    assert len(json.dumps(record, separators=(",", ":")).encode()) <= 4096


def test_serialization_preserves_failure_visibility_instead_of_widening_delivery():
    original = diagnostic(visibility="diagnostic_only")
    assert serialize_diagnostics(original)["visibility"] == "diagnostic_only"
    assert sanitize_diagnostics(original).visibility == original.visibility


def test_failure_emission_is_independent_of_sink_errors_not_cancellation():
    class BrokenLogger:
        def __init__(self, failure):
            self.failure = failure

        def log(self, *_args, **_kwargs):
            raise self.failure

    original = diagnostic()
    emit_failure(BrokenLogger(OSError("PRIVATE_SECRET")), original)
    assert serialize_diagnostics(original)["diagnostic_id"] == original.diagnostic_id
    with pytest.raises(asyncio.CancelledError):
        emit_failure(BrokenLogger(asyncio.CancelledError()), original)


def test_exception_builder_retains_attached_causes_and_native_group_evidence():
    attached = diagnostic(status_code=403, fields={"provider": "openrouter"})
    provider_error = RuntimeError("PRIVATE_SECRET")
    provider_error.failure_diagnostics = attached
    native = PermissionError(13, "PRIVATE_SECRET", "C:/Users/private/file")
    original = ExceptionGroup("PRIVATE_SECRET", [provider_error, native])
    value = serialize_diagnostics(
        exception_diagnostics(
            original,
            component="application_control",
            operation="dispatch",
            code="application_operation_failed",
        )
    )

    assert value["code"] == "application_operation_failed"
    assert value["causes"][0]["causes"][0]["diagnostic_id"] == attached.diagnostic_id
    assert value["causes"][0]["causes"][0]["status_code"] == 403
    assert value["causes"][1]["fields"]["errno"] == 13
    assert "PRIVATE_SECRET" not in json.dumps(value)
    assert "C:/Users" not in json.dumps(value)


def test_exception_builder_bounds_cyclic_chains_without_reinterpreting_original():
    original = ValueError("PRIVATE_SECRET")
    original.__cause__ = original
    safe = exception_diagnostics(
        original,
        component="application_control",
        operation="dispatch",
        code="application_operation_failed",
    )
    assert safe.causes_truncated is True
    assert safe.fields["exception_class"] == "ValueError"


def test_probe_returns_original_failure_when_diagnostic_sink_raises(monkeypatch):
    import logging

    from puripuly_heart.providers.llm.error_details import LLMResponseError, record_probe_failure

    original = LLMResponseError("PRIVATE_SECRET", provider="qwen", status_code=403)

    def reject(*_args, **_kwargs):
        raise OSError("PRIVATE_SECRET")

    monkeypatch.setattr(
        logging.getLogger("puripuly_heart.providers.llm.error_details"), "log", reject
    )
    retained = record_probe_failure(original, provider="qwen")
    assert retained.diagnostic_id == original.failure_diagnostics.diagnostic_id
    assert retained.status_code == 403
    assert retained.category == "auth"
    assert "PRIVATE_SECRET" not in repr(retained)


@pytest.mark.parametrize("shape", sorted(TRANSPORT_ENUM_FIELDS["response_shape"]))
def test_canonical_response_shapes_survive_retained_and_projected_failure_views(shape):
    original = diagnostic(
        component="provider.llm",
        operation="probe",
        code="provider.invalid_response",
        category="invalid_response",
        status_code=200,
        operation_id=uuid4().hex,
        request_id=uuid4().hex,
        fields={"response_shape": shape},
    )
    history = DiagnosticHistory()
    history.bind_instance("response-shape-test")
    history.retain(original, severity="error", disposition="failed", stage="probe")
    retained = history.query({"diagnostic_id": original.diagnostic_id})["records"][0]
    projected = project_diagnostic_record(retained)
    serialized = serialize_diagnostics(original)

    for value in (serialized, retained, projected):
        assert value["fields"] == {"response_shape": shape}
        assert value["diagnostic_id"] == original.diagnostic_id
        assert value["operation_id"] == original.operation_id
        assert value["request_id"] == original.request_id
        assert value["status_code"] == 200
        assert value["code"] == original.code
    assert serialized["rejected_fields"] == 0
    for record in (retained, projected):
        assert record["diagnostic_sequence"] == 1
        assert record["detail"]["rejected_fields"] == 0
        assert record["detail"]["state"] == "available"
        assert record["disposition"] == "failed"


@pytest.mark.parametrize(
    "shape",
    [
        "PRIVATE_RESPONSE_SHAPE token=synthetic-secret C:/Users/private/file",
        ["public_error_envelope", "PRIVATE_RESPONSE_SHAPE"],
        {"shape": "PRIVATE_RESPONSE_SHAPE"},
        True,
        403,
    ],
)
def test_unrecognized_response_shapes_are_rejected_without_erasing_failure_identity(shape):
    original = diagnostic(fields={"response_shape": shape, "errno": 13})
    history = DiagnosticHistory()
    history.bind_instance("response-shape-test")
    history.retain(original, severity="error", disposition="failed")
    retained = history.query({"diagnostic_id": original.diagnostic_id})["records"][0]
    projected = project_diagnostic_record(retained)
    serialized = serialize_diagnostics(original)

    for value in (serialized, retained, projected):
        assert value["fields"] == {"errno": 13}
        assert value["diagnostic_id"] == original.diagnostic_id
        assert value["code"] == original.code
        text = json.dumps(value)
        assert "PRIVATE_RESPONSE_SHAPE" not in text
        assert "synthetic-secret" not in text
        assert "C:/Users" not in text
    assert serialized["rejected_fields"] == 1
    for record in (retained, projected):
        assert record["detail"]["rejected_fields"] == 1
        assert record["detail"]["state"] == "rejected"
