from __future__ import annotations

import json

from puripuly_heart.core.overlay.diagnostics import OverlayDiagnosticsRecorder


def test_overlay_failure_jsonl_redacts_child_output_and_summary_fields(tmp_path) -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-redaction-test",
        diagnostics_dir=tmp_path,
    )
    recorder.record_child_line(
        "stderr",
        "provider_response_body={'error':'bad','token':'provider-secret-jsonl'}",
    )

    path = recorder.dump_failure(
        failure_reason="runtime_crashed",
        broker_raw_message="eligibility failed token=broker-secret-jsonl",
        local_llm_extra_body="{'authorization':'Bearer local-secret-jsonl'}",
        file_contents="private document contents",
        raw_exception="RuntimeError('raw provider exception')",
        stack_trace='File "provider.py", line 42, in translate',
    )

    raw_dump = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw_dump.splitlines()]
    assert rows[0]["event"] == "failure_summary"
    assert "provider-secret-jsonl" not in raw_dump
    assert "broker-secret-jsonl" not in raw_dump
    assert "local-secret-jsonl" not in raw_dump
    assert "provider_response_body" not in raw_dump
    assert "broker_raw_message" not in raw_dump
    assert "local_llm_extra_body" not in raw_dump
    assert "file_contents" not in raw_dump
    assert "private document contents" not in raw_dump
    assert "raw_exception" not in raw_dump
    assert "raw provider exception" not in raw_dump
    assert "stack_trace" not in raw_dump
    assert 'File "provider.py"' not in raw_dump
    assert "[provider-response-body-redacted]" in raw_dump
    assert "[broker-raw-message-redacted]" in raw_dump
    assert "[local-llm-extra-body-redacted]" in raw_dump
    assert "[redacted]" in raw_dump


def test_overlay_failure_jsonl_redacts_token_assignment_variants(tmp_path) -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-token-variant-redaction-test",
        diagnostics_dir=tmp_path,
    )
    recorder.record_child_line(
        "stderr",
        "provider failed access_token=jsonl-access-secret refreshToken=jsonl-refresh-secret",
    )

    path = recorder.dump_failure(
        failure_reason="runtime_crashed",
        id_token="jsonl-structured-id-secret",
        summary="broker failed idToken=jsonl-id-secret authToken=jsonl-auth-secret",
    )

    raw_dump = path.read_text(encoding="utf-8")
    assert "jsonl-access-secret" not in raw_dump
    assert "jsonl-refresh-secret" not in raw_dump
    assert "jsonl-structured-id-secret" not in raw_dump
    assert "jsonl-id-secret" not in raw_dump
    assert "jsonl-auth-secret" not in raw_dump
    assert "[redacted]" in raw_dump


def test_overlay_process_trace_is_monotonic_sanitized_and_included_in_failure_dump(
    tmp_path,
) -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-trace-test",
        diagnostics_dir=tmp_path,
    )

    event = recorder.record_process(
        "overlay_trace",
        trace_event="bounds_confirmed",
        generation=3,
        monotonic_ms=12.5,
        canonical_bounds={"x": 10, "y": 20, "width": 800, "height": 240},
        subtitle_content="private subtitle text",
    )

    assert event["monotonic_ms"] >= 0
    assert event["source_monotonic_ms"] == 12.5
    assert event["generation"] == 3
    assert event["canonical_bounds"] == {"x": 10, "y": 20, "width": 800, "height": 240}
    assert "subtitle_content" not in event
    assert "private subtitle text" not in json.dumps(event)

    raw_dump = recorder.dump_failure(failure_reason="startup_timeout").read_text(encoding="utf-8")
    assert '"trace_event": "bounds_confirmed"' in raw_dump
    assert "private subtitle text" not in raw_dump
