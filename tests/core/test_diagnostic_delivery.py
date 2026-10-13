from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
import time
from uuid import uuid4

import pytest

from puripuly_heart.core.diagnostics import (
    DIAGNOSTIC_PROJECTION_ATTRIBUTE,
    DIAGNOSTIC_RECORD_ATTRIBUTE,
    emit_failure,
)
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.runtime_logging import SessionRuntimeLoggingService, configure_main_logging


def failure(**changes):
    values = dict(
        component="settings_repository",
        operation="save",
        code="settings_save_failed",
        category="transaction",
        visibility="basic",
        content_policy="metadata_only",
        status_code=None,
        retry_after_ms=None,
        fields={"errno": 13},
    )
    values.update(changes)
    return ErrorDiagnostics(**values)


def logging_service(tmp_path):
    root = logging.getLogger(f"test.diagnostic-delivery.{uuid4().hex}")
    root.propagate = False
    sinks = configure_main_logging(root_logger=root, log_dir=tmp_path)
    basic = io.StringIO()
    sinks.stream_handler.setStream(basic)
    service = SessionRuntimeLoggingService(root_logger=root, sinks=sinks)
    service.configure_diagnostics(instance_id=str(uuid4()))
    return root, sinks, service, basic


def finish(root, sinks, service):
    service.close()
    sinks.close(force=True)
    root.handlers.clear()


def test_root_typed_ingress_matches_history_file_and_basic_once(tmp_path):
    root, sinks, service, basic = logging_service(tmp_path)
    events = []
    service.configure_diagnostics(event_sink=events.append)
    original = failure(
        operation_id=uuid4().hex, fields={"message": "FORBIDDEN_DIAGNOSTIC_SENTINEL", "errno": 13}
    )
    try:
        emit_failure(
            root,
            original,
            stage="settings_save",
            exception=PermissionError(13, "FORBIDDEN_EXCEPTION_SENTINEL"),
        )
        service.emit_failure(original, stage="settings_save")
        records = service.diagnostic_history({})["records"]
        assert len(records) == len(events) == 1
        assert records[0] == events[0]["diagnostic"]
        assert records[0]["diagnostic_id"] == original.diagnostic_id
        assert records[0]["operation_id"] == original.operation_id
        assert records[0]["fields"]["errno"] == 13
        assert records[0]["detail"]["state"] == "rejected"
        assert basic.getvalue().count("settings_save_failed") == 1
        assert original.diagnostic_id not in basic.getvalue()
    finally:
        finish(root, sinks, service)
    text = sinks.log_file.read_text(encoding="utf-8")
    assert text.count(original.diagnostic_id) == 1
    selected = [
        json.loads(line.partition("[Failure] ")[2])
        for line in text.splitlines()
        if "[Failure] {" in line
    ]
    assert selected == records
    assert "FORBIDDEN" not in text


def test_untyped_exception_and_forged_child_projection_remain_fail_closed(tmp_path):
    root, sinks, service, _ = logging_service(tmp_path)
    try:
        root.error(
            "FORBIDDEN_FREEFORM_SENTINEL",
            exc_info=(ValueError, ValueError("FORBIDDEN_CHAIN_SENTINEL"), None),
            extra={DIAGNOSTIC_PROJECTION_ATTRIBUTE: {"message": "FORBIDDEN_CHILD_SENTINEL"}},
        )
        assert service.diagnostic_history({})["records"] == []
        root.error(
            "FORBIDDEN_INVALID_RECORD_SENTINEL",
            extra={DIAGNOSTIC_RECORD_ATTRIBUTE: {"arbitrary_child": "FORBIDDEN"}},
        )
        assert service.diagnostic_health()["history"]["rejected_records"] == 1
    finally:
        finish(root, sinks, service)
    text = sinks.log_file.read_text(encoding="utf-8")
    assert "FORBIDDEN" not in text
    assert "untrusted_record_redacted" in text


def test_failed_file_destination_does_not_destroy_live_health_or_retention(tmp_path, monkeypatch):
    root, sinks, service, _ = logging_service(tmp_path)
    delivered = threading.Event()

    def denied(_pending):
        delivered.set()
        raise PermissionError(13, "FORBIDDEN_FILE_SENTINEL")

    monkeypatch.setattr(sinks.file_handler, "_write_pending", denied)
    try:
        record = service.emit_failure(failure())
        assert delivered.wait(2)
        deadline = time.monotonic() + 2
        while (
            not service.diagnostic_health()["file"]["delivery_failures"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        health = service.diagnostic_health()
        assert health["file"]["availability"] == "failed"
        assert health["file"]["delivery_failures"] > 0
        assert health["file"]["persistence"] == "incomplete"
        assert service.diagnostic_history({"diagnostic_id": record["diagnostic_id"]})[
            "records"
        ] == [record]
        cause_id = health["file"]["delivery_diagnostic_id"]
        cause = service.diagnostic_history({"diagnostic_id": cause_id})["records"][0]
        assert cause["code"] == "log_file_delivery_failed"
        assert cause["file_evidence"] == "unavailable"
        assert cause["fields"]["errno"] == 13
        assert cause["fields"]["exception_class"] == "PermissionError"
        assert "FORBIDDEN" not in json.dumps(health)
    finally:
        service.close()
        with pytest.raises(RuntimeError, match="file delivery failed"):
            sinks.close(force=True)
        root.handlers.clear()


def test_stalled_writer_overflow_is_nonblocking_and_shutdown_loss_is_visible(tmp_path, monkeypatch):
    root, sinks, service, _ = logging_service(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = sinks.file_handler._write_pending

    def slow(pending):
        entered.set()
        release.wait(10)
        original(pending)

    monkeypatch.setattr(sinks.file_handler, "_write_pending", slow)
    try:
        service.emit_failure(failure())
        assert entered.wait(2)
        start = time.monotonic()
        for _ in range(2200):
            service.emit_failure(failure(visibility="diagnostic_only"))
        assert time.monotonic() - start < 5
        health = service.diagnostic_health()
        assert health["file"]["queue_pending"] > 0
        assert health["file"]["dropped_records"] > 0
        assert health["file"]["written_records"] == 0
        assert health["history"]["evicted_records"] > 0
        closed_at = time.monotonic()
        with pytest.raises(TimeoutError, match="listener stop timed out"):
            sinks.close(force=True)
        assert time.monotonic() - closed_at < 5
        after = service.diagnostic_health()
        assert after["file"]["shutdown_complete"] is False
        assert after["file"]["persistence"] == "incomplete"
    finally:
        release.set()
        finish(root, sinks, service)


@pytest.mark.asyncio
async def test_threaded_producer_and_failed_subscriber_keep_independent_history(tmp_path):
    root, sinks, service, _ = logging_service(tmp_path)
    delivered = asyncio.Event()

    def broken(_event):
        delivered.set()
        raise RuntimeError("FORBIDDEN_CONSUMER_SENTINEL")

    service.configure_diagnostics(event_sink=broken)
    try:
        result = await asyncio.to_thread(service.emit_failure, failure())
        await asyncio.wait_for(delivered.wait(), 2)
        assert service.diagnostic_history({})["records"] == [result]
        assert service.diagnostic_health()["history"]["subscriber_delivery_failures"] == 1
    finally:
        finish(root, sinks, service)


def test_throwing_lazy_builder_retains_original_safe_cause_once(tmp_path):
    from puripuly_heart.core.runtime.logging import RuntimeLoggingService

    root, sinks, session, _ = logging_service(tmp_path)
    owner = RuntimeLoggingService(session_service=session)
    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        raise PermissionError(13, "PRIVATE_BUILDER_EXCEPTION_MARKER")

    try:
        assert owner.emit_diagnostic_lazy(builder) is False
        assert calls == 1
        record = session.diagnostic_history({})["records"][0]
        assert record["code"] == "diagnostic_builder_failed"
        assert record["fields"]["exception_class"] == "PermissionError"
        assert record["fields"]["errno"] == 13
        assert "PRIVATE_BUILDER" not in json.dumps(record)
    finally:
        finish(root, sinks, session)
