from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from puripuly_heart.core.overlay import diagnostics as diagnostics_module
from puripuly_heart.core.overlay.diagnostics import OverlayDiagnosticsRecorder


async def _dump_path(
    recorder: OverlayDiagnosticsRecorder, *, outcome: str = "failure", **fields: object
):
    receipt = await recorder.dump_evidence(outcome=outcome, **fields)
    assert receipt["outcome"] == "written"
    assert recorder.last_dump_path is not None
    return recorder.last_dump_path


@pytest.mark.asyncio
async def test_overlay_failure_jsonl_redacts_child_output_and_summary_fields(
    tmp_path,
) -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-redaction-test",
        diagnostics_dir=tmp_path,
    )
    recorder.record_child_line(
        "stderr",
        "provider_response_body={'error':'bad','token':'provider-secret-jsonl'}",
    )

    path = await _dump_path(
        recorder,
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


@pytest.mark.asyncio
async def test_overlay_failure_jsonl_redacts_token_assignment_variants(tmp_path) -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-token-variant-redaction-test",
        diagnostics_dir=tmp_path,
    )
    recorder.record_child_line(
        "stderr",
        "provider failed access_token=jsonl-access-secret refreshToken=jsonl-refresh-secret",
    )

    path = await _dump_path(
        recorder,
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


@pytest.mark.asyncio
async def test_overlay_process_trace_is_monotonic_sanitized_and_included_in_failure_dump(
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

    raw_dump = (await _dump_path(recorder, failure_reason="startup_timeout")).read_text(
        encoding="utf-8"
    )
    assert '"trace_event": "bounds_confirmed"' in raw_dump
    assert "private subtitle text" not in raw_dump


@pytest.mark.asyncio
async def test_overlay_presenter_bridge_translation_events_are_recorded_and_dumped(
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

    raw_dump = (await _dump_path(recorder, failure_reason="runtime_crashed")).read_text(
        encoding="utf-8"
    )
    assert '"event": "snapshot_publish"' in raw_dump
    assert '"event": "entry_removed"' in raw_dump
    assert '"event": "broadcast_finish"' in raw_dump
    assert '"event": "overlay_emit"' in raw_dump
    assert "private overlay text" not in raw_dump
    assert "private transcript text" not in raw_dump


@pytest.mark.asyncio
async def test_overlay_chatbox_stt_and_native_stages_are_dumped_without_payload_text(
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
    raw_dump = (await _dump_path(recorder, failure_reason="runtime_crashed")).read_text(
        encoding="utf-8"
    )
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


def test_native_full_batch_preserves_safe_correlation_fields() -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-native-batch",
        logging_mode="detailed",
    )
    records = [
        {
            "sequence": index,
            "logical_revision": 9 + index,
            "scene_generation": 2,
            "logical_causes": ["final"],
            "render_generation": index,
            "submission_attempt": index,
            "stage": "handoff",
            "outcome": "submitted",
            "visibility": "requested_visible",
            "observed_at_ms": 1000 + index,
            "reason": "quiet_tail",
            "lease_disposition": "valid",
            "handoff_mode": "cached_frame_rehandoff",
            "content_identity": f"digest-{index}",
            "dropped_unacknowledged_records": 0,
            "logger_dropped_records": 0,
            "caption": "must not be retained",
            "path": "C:/private/user/file",
        }
        for index in range(8)
    ]

    assert recorder.ingest_native_child_line("presentation_diagnostics " + json.dumps(records))

    assert len(recorder.native_events) == 8
    assert recorder.native_events[0]["native_sequence"] == 0
    assert recorder.native_events[-1]["logical_revision"] == 16
    assert recorder.native_events[-1]["handoff_mode"] == "cached_frame_rehandoff"
    serialized = json.dumps(list(recorder.native_events))
    assert "must not be retained" not in serialized
    assert "C:/private" not in serialized


def test_native_evidence_distinguishes_real_render_from_successful_rehandoff() -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-evidence",
        logging_mode="detailed",
    )
    recorder.ingest_native_child_line(
        "presentation_diagnostics "
        + json.dumps(
            [
                {
                    "stage": "render_returned",
                    "outcome": "success",
                    "handoff_mode": "off",
                    "reason": "fresh_render_required",
                    "render_generation": 4,
                },
                {
                    "stage": "submission_returned",
                    "outcome": "success",
                    "handoff_mode": "cached_frame_rehandoff",
                    "reason": "cached_completed_frame_rehandoff",
                    "render_generation": 4,
                    "submission_attempt": 5,
                    "content_identity": "digest",
                },
            ]
        )
    )

    summary = recorder.evidence_summary()
    assert summary["real_render_observed"] is True
    assert summary["cache_hit_observed"] is True
    assert summary["cached_frame_rehandoff_observed"] is True


def test_measurement_checkpoint_retains_early_phase_after_native_ring_rollover() -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-phase-retention",
        logging_mode="detailed",
    )
    for sequence in range(8):
        recorder.ingest_native_child_line(
            "presentation_diagnostics "
            + json.dumps(
                [
                    {
                        "sequence": sequence,
                        "logical_revision": 4,
                        "stage": "handoff",
                        "outcome": "submitted",
                        "handoff_mode": "off",
                        "dropped_unacknowledged_records": 0,
                        "logger_dropped_records": 0,
                    }
                ]
            )
        )
    recorder.capture_measurement_phase("self_m1_translation", scene_revision=4)
    for sequence in range(100):
        recorder.ingest_native_child_line(
            "presentation_diagnostics "
            + json.dumps([{"sequence": 100 + sequence, "stage": "idle", "outcome": "observed"}])
        )

    retained = [
        event
        for event in recorder.measurement_phase_events
        if event.get("phase") == "self_m1_translation"
    ]
    checkpoint = next(event for event in retained if event["event"] == "checkpoint")
    assert any(event.get("logical_revision") == 4 for event in retained)
    assert checkpoint["native_correlation"] == "observed"
    assert checkpoint["native_records_unavailable"] == 0
    assert checkpoint["native_records_omitted"] == 0
    assert recorder.evidence_summary()["memory_dropped"]["native"] == 58


@pytest.mark.asyncio
async def test_repeated_native_loss_samples_use_high_water_and_preclude_complete_evidence(
    tmp_path,
) -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-native-loss",
        diagnostics_dir=tmp_path,
        logging_mode="detailed",
    )
    for sequence, presenter_loss, logger_loss in (
        (1, 11, 2),
        (2, 11, 2),
        (3, 12, 2),
    ):
        recorder.ingest_native_child_line(
            "presentation_diagnostics "
            + json.dumps(
                [
                    {
                        "sequence": sequence,
                        "stage": "submission_returned",
                        "outcome": "success",
                        "dropped_unacknowledged_records": presenter_loss,
                        "logger_dropped_records": logger_loss,
                    }
                ]
            )
        )

    recorder.capture_measurement_phase("self_m1_translation", scene_revision=4)
    checkpoint = next(
        event for event in recorder.measurement_phase_events if event["event"] == "checkpoint"
    )
    retained = [
        event
        for event in recorder.measurement_phase_events
        if event.get("category") == "measurement_phase_native"
    ]

    assert checkpoint["native_correlation"] == "partial"
    assert checkpoint["native_loss"]["dropped_unacknowledged_records"] == {
        "state": "observed",
        "high_water": 12,
        "delta": 12,
        "continuity_gaps": 0,
        "continuity_gaps_total": 0,
        "missing_samples": 0,
        "missing_samples_total": 0,
    }
    assert checkpoint["native_loss"]["logger_dropped_records"]["high_water"] == 2
    assert [event["dropped_unacknowledged_records_delta"] for event in retained] == [11, 0, 1]
    assert [event["logger_dropped_records_delta"] for event in retained] == [2, 0, 0]

    receipt = await recorder.dump_evidence(outcome="success")
    assert receipt["outcome"] == "written"
    assert receipt["complete"] is False
    assert receipt["native_known_loss"] is True
    assert receipt["native_terminal_delivery_completeness"] == "unknown"
    assert receipt["native_loss_counters"]["dropped_unacknowledged_records"]["high_water"] == 12


@pytest.mark.asyncio
async def test_absent_native_loss_samples_are_unknown_not_zero(tmp_path) -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-native-loss-unknown",
        diagnostics_dir=tmp_path,
        logging_mode="detailed",
    )
    recorder.ingest_native_child_line(
        'presentation_diagnostics [{"sequence":1,"stage":"submission_returned"}]'
    )
    recorder.capture_measurement_phase("self_m1_translation", scene_revision=4)

    summary = recorder.evidence_summary()
    checkpoint = next(
        event for event in recorder.measurement_phase_events if event["event"] == "checkpoint"
    )
    receipt = await recorder.dump_evidence(outcome="success")

    assert summary["native_counter_state_unknown"] is True
    assert summary["native_loss_counters"]["logger_dropped_records"]["high_water"] is None
    assert checkpoint["native_correlation"] == "partial"
    assert checkpoint["native_loss"]["logger_dropped_records"]["state"] == "unknown"
    assert receipt["complete"] is False
    assert receipt["native_evidence_completeness"] == "incomplete"


def test_phase_checkpoint_reports_partial_when_early_records_precede_last_eight() -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-phase-partial",
        logging_mode="detailed",
    )
    for sequence in range(12):
        recorder.ingest_native_child_line(
            "presentation_diagnostics "
            + json.dumps(
                [
                    {
                        "sequence": sequence,
                        "logical_revision": 4,
                        "stage": ("visibility_observed" if sequence < 4 else "submission_returned"),
                        "outcome": "success",
                    }
                ]
            )
        )

    recorder.capture_measurement_phase("self_m1_translation", scene_revision=4)
    checkpoint = next(
        event for event in recorder.measurement_phase_events if event["event"] == "checkpoint"
    )

    assert checkpoint["native_record_count"] == 8
    assert checkpoint["native_records_omitted"] == 4
    assert checkpoint["native_records_unavailable"] == 0
    assert checkpoint["native_correlation"] == "partial"
    assert not any(
        event.get("native_event") == "visibility_observed"
        for event in recorder.measurement_phase_events
    )


@pytest.mark.asyncio
async def test_dump_enforces_line_and_file_bounds_and_reports_loss(tmp_path) -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-bounds",
        diagnostics_dir=tmp_path,
        logging_mode="detailed",
    )
    for index in range(300):
        recorder.record_process("flood", index=index, safe_metadata="x" * 5000)

    path = await _dump_path(recorder, failure_reason="runtime_crashed")
    raw = path.read_bytes()
    rows = raw.splitlines(keepends=True)
    summary = json.loads(rows[0])

    assert len(raw) <= 1024 * 1024
    assert all(len(row) <= 4 * 1024 for row in rows)
    assert summary["records_truncated"] > 0
    assert summary["memory_dropped"]["process"] == 44
    assert summary["complete"] is False


@pytest.mark.asyncio
async def test_dump_deadline_is_abandoned_without_blocking_caller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-deadline",
        diagnostics_dir=tmp_path,
    )
    original = diagnostics_module.OverlayDiagnosticsRecorder._write_dump_file

    def slow_write(temporary, path, content):
        time.sleep(0.05)
        original(temporary, path, content)

    monkeypatch.setattr(diagnostics_module, "_DIAGNOSTIC_DUMP_DEADLINE_SECONDS", 0.01)
    monkeypatch.setattr(
        diagnostics_module.OverlayDiagnosticsRecorder,
        "_write_dump_file",
        staticmethod(slow_write),
    )
    started = time.monotonic()
    receipt = await recorder.dump_evidence(outcome="failure")

    assert time.monotonic() - started < 0.04
    assert receipt["outcome"] == "abandoned"
    assert receipt["dump_abandoned"] == 1
    first_receipt = dict(recorder.last_dump_receipt or {})
    second = await recorder.dump_evidence(outcome="failure")
    await asyncio.sleep(0.06)
    assert second["reason"] == "writer_disabled_after_abandonment"
    assert recorder.last_dump_path is None
    assert recorder.last_dump_receipt == second
    assert first_receipt["reason"] == "dump_deadline_exceeded"


def test_blocked_diagnostic_writer_does_not_delay_subprocess_termination(tmp_path) -> None:
    probe = tmp_path / "blocked_writer_probe.py"
    probe.write_text(
        """import asyncio
import json
import time
from pathlib import Path
from puripuly_heart.core.overlay import diagnostics as module
from puripuly_heart.core.overlay.diagnostics import OverlayDiagnosticsRecorder

module._DIAGNOSTIC_DUMP_DEADLINE_SECONDS = 0.05
def blocked_writer(temporary, path, content):
    time.sleep(30)
OverlayDiagnosticsRecorder._write_dump_file = staticmethod(blocked_writer)

async def main():
    recorder = OverlayDiagnosticsRecorder('blocked-writer', diagnostics_dir=Path.cwd())
    started = time.monotonic()
    first = await recorder.dump_evidence(outcome='failure')
    second = await recorder.dump_evidence(outcome='failure')
    print(json.dumps({
        'elapsed': time.monotonic() - started,
        'first': first,
        'second': second,
        'last_dump_path': recorder.last_dump_path,
    }))

asyncio.run(main())
""",
        encoding="utf-8",
    )
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, str(probe)],
        cwd=Path(diagnostics_module.__file__).resolve().parents[4],
        text=True,
        capture_output=True,
        timeout=1.0,
        check=True,
    )
    elapsed = time.monotonic() - started
    result = json.loads(completed.stdout)

    assert elapsed < 1.0
    assert result["elapsed"] < 0.2
    assert result["first"]["reason"] == "dump_deadline_exceeded"
    assert result["second"]["reason"] == "writer_disabled_after_abandonment"
    assert result["last_dump_path"] is None
