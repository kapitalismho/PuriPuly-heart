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


def test_overlay_presenter_bridge_translation_events_are_recorded_and_dumped(
    tmp_path,
) -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-stage-trace-test",
        diagnostics_dir=tmp_path,
        logging_mode="detailed",
    )

    presenter_event = recorder.record_presenter(
        "snapshot_publish",
        revision=4,
        block_count=1,
        text="private overlay text",
    )
    removal_event = recorder.record_presenter_removal(
        entry_key="self:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    )
    bridge_event = recorder.record_bridge(
        "broadcast_finish",
        revision=4,
        elapsed_ms=12,
        transcript="private transcript text",
    )
    translation_event = recorder.record_translation(
        "overlay_emit",
        event_kind="translation",
        utterance_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        secondary_len=18,
    )

    assert presenter_event["category"] == "presenter"
    assert presenter_event["revision"] == 4
    assert removal_event["category"] == "presenter_removal"
    assert bridge_event["category"] == "bridge"
    assert translation_event["category"] == "translation"
    assert "private overlay text" not in json.dumps(presenter_event)
    assert "private transcript text" not in json.dumps(bridge_event)

    raw_dump = recorder.dump_failure(failure_reason="runtime_crashed").read_text(encoding="utf-8")
    assert '"event": "snapshot_publish"' in raw_dump
    assert '"event": "entry_removed"' in raw_dump
    assert '"event": "broadcast_finish"' in raw_dump
    assert '"event": "overlay_emit"' in raw_dump
    assert "private overlay text" not in raw_dump
    assert "private transcript text" not in raw_dump


def test_overlay_chatbox_stt_and_native_stages_are_dumped_without_payload_text(
    tmp_path,
) -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-gate0-trace-test",
        diagnostics_dir=tmp_path,
        logging_mode="detailed",
    )
    recorder.record_chatbox(
        "page_send",
        utterance_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        page_index=1,
        pending_messages=1,
        oldest_age_s=6.0,
        text="secret chatbox page",
    )
    recorder.record_stt(
        "stt_enqueue",
        channel="self",
        queue_depth=2,
        oldest_age_s=1.5,
        transcript="secret stt text",
    )
    ingested = recorder.ingest_native_child_line(
        'presentation_diagnostics [{"stage":"readiness_observed","logical_revision":9,'
        '"outcome":"timed_out","readiness_us":51000,"observed_runtime_visible":true,'
        '"desired_visible":true,"physical_hmd_visibility":"not_observable"}]'
    )

    assert ingested is True
    raw_dump = recorder.dump_failure(failure_reason="runtime_crashed").read_text(encoding="utf-8")
    assert '"category": "chatbox"' in raw_dump
    assert '"event": "page_send"' in raw_dump
    assert '"category": "stt"' in raw_dump
    assert '"event": "stt_enqueue"' in raw_dump
    assert '"category": "native"' in raw_dump
    assert '"event": "readiness_observed"' in raw_dump
    assert '"actual_visibility": "not_queried"' in raw_dump
    assert "secret chatbox page" not in raw_dump
    assert "secret stt text" not in raw_dump


def test_overlay_stage_memory_is_recorded_only_in_detailed_mode(tmp_path) -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-stage-mode-test",
        diagnostics_dir=tmp_path,
    )

    assert recorder.record_presenter("snapshot_publish", revision=1) == {}
    assert recorder.record_bridge("send_start", revision=1) == {}
    assert recorder.record_translation("overlay_emit") == {}
    assert recorder.record_chatbox("page_send") == {}
    assert recorder.record_stt("stt_enqueue") == {}
    assert (
        recorder.ingest_native_child_line(
            'presentation_diagnostics [{"stage":"readiness_observed"}]'
        )
        is False
    )
    assert list(recorder.presenter_events) == []
    assert list(recorder.bridge_events) == []
    assert list(recorder.translation_events) == []
    assert list(recorder.chatbox_events) == []
    assert list(recorder.stt_events) == []
    assert list(recorder.native_events) == []

    recorder.set_logging_mode("detailed")
    recorder.record_presenter("snapshot_publish", revision=2)
    recorder.record_process("overlay_trace", trace_event="bounds_confirmed")
    assert [event["event"] for event in recorder.presenter_events] == ["snapshot_publish"]
    assert [event["event"] for event in recorder.process_events] == ["overlay_trace"]

    recorder.set_logging_mode("basic")
    assert list(recorder.presenter_events) == []
    assert [event["event"] for event in recorder.process_events] == ["overlay_trace"]
