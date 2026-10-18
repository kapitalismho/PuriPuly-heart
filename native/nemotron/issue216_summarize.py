"""Summarize retained live data; no fresh model/server calls or invented outcomes."""

import argparse
import json
from pathlib import Path

from issue216_smoke import save


def operation_timings(case):
    trace, start = case["trace"], case["playback_started_at"]
    submits = [item for item in trace if item["event"] == "native_submit"]
    controls = [
        item
        for item in trace
        if item["event"] == "control_written" and item["control"] == "Finalize"
    ]
    raw = [
        item
        for item in trace
        if item["event"] == "raw_wire_result" and item["value"].get("type") == "Results"
    ]
    seals, admissions = [], []
    for submit, control in zip(submits, controls):
        turn = submit["identity"]["provider_turn_id"]
        ending = next(
            item
            for item in trace
            if item["event"] == "vad"
            and item["value"]["event"].get("reason") == submit["seal_reason"]
            and item["value"]["segment"]["identity"]["segment_id"]
            == submit["identity"]["segment"]["segment_id"]
        )
        receipt = next(item for item in raw if item["at"] >= control["at"])
        terminal = next(
            item
            for item in trace
            if item["event"] == "native_result"
            and item["value"].get("outcome")
            and item["value"]["identity"]["provider_turn_id"] == turn
        )
        normalized = next(
            item
            for item in trace
            if item["event"] == "normalized_provider_result"
            and item["value"].get("outcome")
            and item["value"]["identity"]["provider_turn_id"] == turn
        )
        successor = next(
            (
                item
                for item in trace
                if item["event"] == "native_begin" and item["at"] > normalized["at"]
            ),
            None,
        )
        ideal_end = submit["sealed_content_ranges"][-1]["source_end_monotonic_s"]
        seals.append(
            {
                "reason": submit["seal_reason"],
                "content_end_sample": submit["sealed_content_ranges"][-1]["normalized_end_sample"],
                "source_capture_end_ideal_at": ideal_end,
                "end_event_after_playback_s": ending["at"] - start,
                "end_event_minus_source_capture_end_s": ending["at"] - ideal_end,
                "end_event_to_submit_s": submit["at"] - ending["at"],
                "submit_to_control_written_s": control["at"] - submit["at"],
                "control_sent_cursor": control["sent_until"],
                "control_to_raw_result_s": receipt["at"] - control["at"],
                "raw_result_to_native_terminal_s": terminal["at"] - receipt["at"],
                "native_terminal_to_normalized_terminal_s": normalized["at"] - terminal["at"],
                "control_to_successor_begin_s": (
                    successor["at"] - control["at"] if successor else None
                ),
            }
        )
    for receipt in raw:
        native = next(
            item
            for item in trace
            if item["event"] == "native_result"
            and item["at"] >= receipt["at"]
            and item["value"].get("source_release")
        )
        contribution_id = native["value"]["contribution"]["contribution_id"]
        children = [
            source
            for source in case["sources"]
            if any(
                origin["contribution_id"] == contribution_id
                for origin in source["source"]["origins"]
            )
        ]
        admissions.append(
            {
                "raw_at": receipt["at"],
                "raw_text": receipt["value"]["channel"]["alternatives"][0]["transcript"],
                "raw_to_native_result_s": native["at"] - receipt["at"],
                "raw_to_first_child_s": min(source["admitted_at"] for source in children)
                - receipt["at"],
                "raw_to_last_child_s": max(source["admitted_at"] for source in children)
                - receipt["at"],
            }
        )
    return {
        "provider_ready_before_playback_s": start - case["provider_session_ready_at"],
        "peak_capture_delivery_lag_s": case["capture_delivery_max_lag_seconds"],
        "seals_and_terminal_progress": seals,
        "raw_to_source_admission": admissions,
        "limits": "Capture peak has no per-frame peak timestamp; these are observed stage intervals, not profiling of internal attribution/socket/GPU/OS time or proof of cause. Same configured timer/reasons do not imply identical actual cut cursor.",
    }


def summary(case):
    sources, trace = case.get("sources", []), case.get("trace", [])
    ends = [
        {
            "at": item["at"],
            "reason": item["value"]["event"]["reason"],
            "source_end_sample": item["value"]["segment"]["content_ranges"][-1][
                "normalized_end_sample"
            ],
        }
        for item in trace
        if item.get("event") == "vad" and "reason" in item.get("value", {}).get("event", {})
    ]
    boundaries = [item for item in trace if item.get("event") == "common_ledger_audio_boundary"]
    ordered = [item["source"]["order"] for item in sources]
    used, overlaps, native_segments = {}, [], []
    for source in sources:
        for origin in source["source"].get("origins", []):
            identity, span = origin.get("contribution_id"), origin.get("contribution_range")
            if not identity or span is None:
                continue
            for previous in used.get(identity, []):
                if max(previous[0], span[0]) < min(previous[1], span[1]):
                    overlaps.append(
                        {"contribution_id": identity, "previous": previous, "current": span}
                    )
            used.setdefault(identity, []).append(span)
    for item in trace:
        event = item.get("value", {})
        if (
            item.get("event") != "native_result"
            or not event.get("source_release")
            or not event.get("contribution")
        ):
            continue
        contribution = event["contribution"]
        children = [
            source
            for source in sources
            if any(
                origin["contribution_id"] == contribution["contribution_id"]
                for origin in source["source"].get("origins", [])
            )
        ]
        native_segments.append(
            {
                "observed_at": item["at"],
                "text": event["text"],
                "contribution": contribution,
                "clocks": event["provenance"]["audio_clocks"],
                "complete_timings": [
                    timing
                    for timing in event["provenance"]["text_timing"]
                    if timing["complete_boundary"]
                ],
                "resulting_children": [
                    {
                        "text": child["source"]["raw_text"],
                        "admitted_at": child["admitted_at"],
                        "speaker": child["source"]["speaker"],
                        "origin_ranges": [
                            origin["contribution_range"]
                            for origin in child["source"]["origins"]
                            if origin["contribution_id"] == contribution["contribution_id"]
                        ],
                    }
                    for child in children
                ],
            }
        )
    first_end = ends[0]["at"] if ends else None
    return {
        "execution_status": case["status"],
        "error": case.get("error"),
        "errors": case.get("errors", []),
        "playback_started_at": case["playback_started_at"],
        "sources": [
            {
                "admitted_at": source["admitted_at"],
                "text": source["source"]["raw_text"],
                "order": source["source"]["order"],
                "speaker": source["source"].get("speaker"),
                "display_group_id": source["source"].get("display_group_id"),
                "contribution_origins": [
                    {
                        "contribution_id": origin["contribution_id"],
                        "contribution_range": origin["contribution_range"],
                        "ownership": origin["ownership"],
                    }
                    for origin in source["source"]["origins"]
                ],
                "prior_context_texts": source.get("prior_context_texts"),
                "lead_before_first_enclosing_end_seconds": (
                    first_end - source["admitted_at"] if first_end is not None else None
                ),
            }
            for source in sources
        ],
        "admitted_text_in_source_order": "".join(
            source["source"]["raw_text"] for source in sources
        ),
        "native_final_segments_and_source_boundaries": native_segments,
        "audio_boundaries_recorded": boundaries,
        "enclosing_ends": ends,
        "controls_written": [item for item in trace if item.get("event") == "control_written"],
        "physical_submissions": [
            {
                "at": item["at"],
                "reason": item["seal_reason"],
                "pcm16_sha256": item["pcm16_sha256"],
                "samples": sum(payload["samples"] for payload in item["payloads"]),
                "context_samples": sum(
                    payload["samples"] for payload in item["payloads"] if payload["context_only"]
                ),
                "declared_source_bytes_match": all(
                    payload["matches_declared_native_input"] for payload in item["payloads"]
                ),
            }
            for item in trace
            if item.get("event") == "native_submit"
        ],
        "original_capture_received_once": case.get("original_capture_received_once"),
        "source_orders_sorted": ordered == sorted(ordered),
        "overlapping_admitted_contribution_ranges": overlaps,
        "trimming_OFF_raw_equals_effective": all(
            source["effective_equals_raw"] for source in sources
        ),
        "actual_audio_samples_written": case["actual_audio_samples_written"],
        "retention_high_water_samples": max(
            (item["value"].get("high_water_samples", 0) for item in case.get("retention", [])),
            default=0,
        ),
        "retention_final": case.get("retention", [])[-1] if case.get("retention") else None,
        "capture_delivery_max_lag_seconds": case.get("capture_delivery_max_lag_seconds"),
        "speaker_final": case.get("speaker_final"),
        "actual_speaker_evidence_count": len(case.get("evidence", [])),
        "acceptance_scope": "Execution/summary is not automatic rich-source acceptance. Complete native grouping, actual source attribution, causal evidence/availability and successor progress must all support that separate claim; assembly annotations never supply speaker evidence.",
    }


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", type=Path)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--timing-reference", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if not args.output.name.startswith("issue216"):
    raise ValueError("summary must remain issue216 task-local")
candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
old = {case["case"]: case for case in baseline["cases"]} if baseline else {}


def normalize(values):
    return {path.replace("\\", "/"): value for path, value in values.items()}


new_hashes = normalize(candidate["execution_source_sha256"])
old_hashes = normalize(baseline["execution_source_sha256"]) if baseline else {}
report = {
    "schema": "issue216-retained-evidence-summary-2",
    "source_reports": [str(path) for path in (args.baseline, args.candidate) if path is not None],
    "execution_source_hashes_same": old_hashes == new_hashes if baseline else None,
    "product_execution_hashes_same": (
        all(
            value == new_hashes.get(path)
            for path, value in old_hashes.items()
            if path.startswith("src/")
        )
        if baseline
        else None
    ),
    "actual_execution_hashes": new_hashes,
    "baseline_contrast": candidate["baseline_contrast"],
    "trimming": "OFF",
    "cases": {},
    "budget_after": candidate["budget_after"],
    "visual_caption_observation": "not_run; source attribution payload is not rendered caption/HMD evidence",
    "translation_calls": 0,
    "profile_selection": "not_run; explicit development0.64 only",
}
for case in candidate["cases"]:
    previous = old.get(case["case"])
    report["cases"][case["case"]] = {
        "identical_original_pcm": (
            previous["input_pcm_f32_sha256"] == case["input_pcm_f32_sha256"] if previous else None
        ),
        "baseline": summary(previous) if previous else {"status": "not_run"},
        "candidate": summary(case),
    }
if args.timing_reference:
    reference = json.loads(args.timing_reference.read_text(encoding="utf-8"))
    reference_cases = {case["case"]: case for case in reference["cases"]}
    report["timing_reference"] = str(args.timing_reference)
    report["operation_timing_comparison"] = {
        case["case"]: {
            "candidate": operation_timings(case),
            "reference": operation_timings(reference_cases[case["case"]]),
        }
        for case in candidate["cases"]
        if case["case"] in reference_cases
    }
save(args.output, report)
print(
    {"output": str(args.output), "cases": list(report["cases"]), "cloud_calls": 0, "model_calls": 0}
)
