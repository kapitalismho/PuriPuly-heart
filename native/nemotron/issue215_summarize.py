"""Summarize issue215_smoke evidence without re-running inference.
Usage: .venv/Scripts/python.exe -B native/nemotron/issue215_summarize.py --baseline ... --candidate ... --output ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def ranges(spans):
    result = []
    for span in spans:
        item = [span["capture_epoch"], span["normalized_start_sample"], span["normalized_end_sample"]]
        if result and result[-1][0] == item[0] and result[-1][2] == item[1]:
            result[-1][2] = item[2]
        else:
            result.append(item)
    return result


def expand(items):
    return {(epoch, sample) for epoch, start, end in items for sample in range(start, end)}


def summarize(case):
    submissions = [item for item in case["trace"] if item["event"] == "native_submit"]
    events = [item for item in case["trace"] if item["event"] == "provider_result"]
    source_units = case["sources"]
    owner_logs = [item for item in case["trace"] if item["event"] == "batch_owner_log"]
    onset = [item for item in case["trace"] if item["event"] == "vad" and "genuine_onset" in item["value"]["event"]]
    seals = [item for item in case["trace"] if item["event"] == "vad" and "reason" in item["value"]["event"]]
    parent_end = {item["value"]["event"]["utterance_id"]: item["value"]["segment"]["sealed_at_monotonic_s"] for item in seals}
    rows = []
    all_source = []
    for item in submissions:
        identity = item["identity"]
        request_id = identity["provider_turn_id"]
        source = ranges(item["sealed_content_ranges"])
        context = ranges([span for payload in item["payloads"] if payload["context_only"] for span in payload["source_ranges"]])
        delivered_new = ranges([span for payload in item["payloads"] if not payload["context_only"] for span in payload["source_ranges"]])
        all_source.extend(delivered_new)
        # Contribution IDs belong to the enclosing result. Exact owned ranges
        # correlate selected native input with the shared source without text,
        # receipt-order guesses or invented word timestamps.
        units = [unit for unit in source_units if any(
            origin["request_id"] == request_id or ranges(origin["source_ranges"]) == source
            for origin in unit["source"]["origins"]
        )]
        logs = [entry["message"] for entry in owner_logs if f"slice={request_id} " in entry["message"]]
        ready = boundary = None
        if logs:
            match = re.search(r"source_boundary=(\S+) evidence_ready=(\S+)", logs[0])
            if match:
                boundary = int(match[1]) if match[1] != "None" else None
                ready = float(match[2]) if match[2] != "None" else None
        row = {
            "native_request_id": request_id,
            "native_pcm16_sha256": item.get("pcm16_sha256"),
            "reason": item["seal_reason"],
            "source_boundary_sample": boundary,
            "source_boundary_seconds": boundary / 16000 if boundary is not None else None,
            "evidence_ready_monotonic_s": ready,
            "submission_monotonic_s": item["at"],
            "submitted_cursor_sample": source[-1][2] if source else None,
            "new_content_ranges": source,
            "actual_delivered_new_ranges": delivered_new,
            "sealed_new_ranges_equal_actual_payloads": source == delivered_new,
            "declared_context_ranges": context,
            "context_samples": sum(end - start for _, start, end in context),
            "enclosing_seal_monotonic_s": parent_end.get(identity["segment"]["segment_id"]),
            "final_source": [
                {
                    "available_at_monotonic_s": min(
                        (origin["source_available_at_monotonic_s"]
                         for origin in unit["source"]["origins"]
                         if origin["source_available_at_monotonic_s"] is not None),
                        default=unit["admitted_at"],
                    ),
                    "admitted_at_monotonic_s": unit["admitted_at"],
                    "order": unit["source"]["order"],
                    "raw_text": unit["source"]["raw_text"],
                    "effective_equals_raw": unit["effective_equals_raw"],
                    "speaker": unit["source"]["speaker"],
                    "origin_attribution": [
                        origin.get("speaker_attribution") for origin in unit["source"]["origins"]
                    ],
                    "synthetic_right_context_samples": [
                        origin.get("synthetic_right_context_samples", 0) for origin in unit["source"]["origins"]
                    ],
                    "synthetic_context_sample_rate_hz": [
                        origin.get("synthetic_context_sample_rate_hz") for origin in unit["source"]["origins"]
                    ],
                    "display_group_id": unit["source"]["display_group_id"],
                    "prior_context_texts": unit["prior_context_texts"],
                }
                for unit in units
            ],
        }
        begun = next(
            entry["request"] for entry in case["trace"]
            if entry["event"] == "native_begin"
            and entry["request"]["identity"]["provider_turn_id"] == request_id
        )
        row["declared_display_group_id"] = begun["display_group_id"]
        row["synthetic_right_context_samples"] = begun.get("synthetic_right_context_samples", 0)
        row["synthetic_context_sample_rate_hz"] = begun.get("synthetic_context_sample_rate_hz")
        row["synthetic_native_input_valid"] = (
            item.get("synthetic_right_context_valid", True)
            and all(payload["matches_declared_native_input"] for payload in item["payloads"])
        )
        row["synthetic_context_metadata_preserved"] = all(
            origin.get("synthetic_right_context_samples", 0) == row["synthetic_right_context_samples"]
            and origin.get("synthetic_context_sample_rate_hz") == row["synthetic_context_sample_rate_hz"]
            for unit in units for origin in unit["source"]["origins"]
        ) if units else None
        native = next((
            entry for entry in case["trace"]
            if entry["event"] == "native_result"
            and entry["value"]["identity"]["provider_turn_id"] == request_id
        ), None)
        row["native_result"] = ({
            "available_at_monotonic_s": native["at"],
            "outcome": native["value"]["outcome"],
            "text": native["value"]["text"],
            "failure_reason": native["value"].get("failure_reason"),
        } if native else None)
        row["native_text_matches_admitted_source"] = (
            "".join(unit["raw_text"] for unit in row["final_source"]) == native["value"]["text"]
            if native and native["value"]["outcome"] in ("final", "empty") else None
        )
        row["native_payload_matches_declared_source"] = (
            all(payload["matches_declared_source_pcm"] for payload in item["payloads"] if payload["source_ranges"])
            if all("matches_declared_source_pcm" in payload for payload in item["payloads"])
            else None
        )
        rows.append(row)
    owned = ranges([span for snapshot in case.get("segments", []) for span in snapshot["content_ranges"]])
    unique = expand(all_source)
    owned_unique = expand(owned)
    source_count = sum(end - start for _, start, end in all_source)
    early = [row for row in rows if row["reason"] == "speaker" and any(unit["available_at_monotonic_s"] < row["enclosing_seal_monotonic_s"] for unit in row["final_source"])]
    unsupported = [row for row in rows if row["reason"] == "speaker" and (row["evidence_ready_monotonic_s"] is None or row["evidence_ready_monotonic_s"] > row["submission_monotonic_s"])]
    # History snapshots repeat earlier entries legitimately; evaluate each snapshot.
    unique_history = all(len({entry["source_id"] for entry in unit["history_after"]}) == len(unit["history_after"]) for unit in source_units)
    group_matches = bool(source_units) and all(
        row["declared_display_group_id"] is not None
        and all(unit["display_group_id"] == row["declared_display_group_id"] for unit in row["final_source"])
        for row in rows
    )
    text_accounting = []
    for item in events:
        event = item["value"]
        if event.get("outcome") not in ("final", "empty"):
            continue
        request_id = event["identity"]["provider_turn_id"]
        admitted_text = "".join(
            unit["source"]["raw_text"] for unit in source_units
            if any(origin["request_id"] == request_id for origin in unit["source"]["origins"])
        )
        text_accounting.append(admitted_text == event["text"])
    return {
        "execution_status": case["status"],
        "error": case.get("error"),
        "source_contract_errors": case.get("source_contract_errors", []),
        "transcript": " ".join(unit["source"]["raw_text"] for unit in source_units),
        "source_count": len(source_units),
        "submissions": rows,
        "checks": {
            "submitted_new_content_exactly_once": "passed" if source_count == len(unique) and unique == owned_unique else "failed",
            "sealed_new_ranges_equal_actual_payloads": "passed" if all(row["sealed_new_ranges_equal_actual_payloads"] for row in rows) else "failed",
            "submitted_new_samples": source_count,
            "owned_new_samples": len(owned_unique),
            "duplicated_new_samples": source_count - len(unique),
            "missing_owned_samples": len(owned_unique - unique),
            "extra_new_samples": len(unique - owned_unique),
            "speaker_cut_causal_evidence": "failed" if unsupported else ("passed" if any(row["reason"] == "speaker" for row in rows) else "not_run"),
            "actual_early_source_before_enclosing_seal": "passed" if early else "not_run",
            "early_source_count": len(early),
            "once_only_context_entries": "passed" if unique_history and source_units else "not_run",
            "raw_effective_trimming_off": "passed" if all(unit["effective_equals_raw"] for unit in source_units) else "failed",
            "verified_display_group_preserved": "passed" if group_matches else "failed",
            "finalized_text_admitted_exactly_once": "passed" if text_accounting and all(text_accounting) else "failed",
            "native_finalized_text_admitted_exactly_once": (
                "passed" if all(row["native_text_matches_admitted_source"] is True for row in rows)
                else "not_run" if any(row["native_result"] is None for row in rows) else "failed"
            ),
            "native_payload_matches_declared_source": (
                "passed" if all(row["native_payload_matches_declared_source"] is True for row in rows)
                else "not_run" if any(row["native_payload_matches_declared_source"] is None for row in rows) else "failed"
            ),
            "synthetic_native_context_input_valid": (
                "failed" if any(row["synthetic_native_input_valid"] is False for row in rows)
                else "passed" if all(row["synthetic_native_input_valid"] is True for row in rows) else "not_run"
            ),
            "synthetic_context_metadata_preserved_to_source_child": (
                "failed" if any(row["synthetic_context_metadata_preserved"] is False for row in rows)
                else "passed" if any(row["synthetic_context_metadata_preserved"] is True for row in rows) else "not_run"
            ),
            "original_capture_received_once": (
                "passed" if case.get("original_capture_received_once") else
                "failed" if "original_capture_received_once" in case else "not_run"
            ),
        },
        "genuine_onset_context_samples": [
            item["value"]["event"]["pre_roll"]["samples"] for item in onset
            if item["value"]["event"]["genuine_onset"]
        ],
        "hard_rollover_context_samples": [
            item["value"]["event"]["pre_roll"]["samples"] for item in onset
            if not item["value"]["event"]["genuine_onset"]
        ],
        "seal_reasons": [item["value"]["event"]["reason"] for item in seals],
        "speaker_final": case.get("speaker_final"),
        "nonfinal_or_empty_results": [
            {"outcome": item["value"].get("outcome"), "text": item["value"].get("text"),
             "failure_reason": item["value"].get("failure_reason")}
            for item in events if item["value"].get("outcome") not in (None, "final")
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    old = {case["case"]: case for case in baseline["cases"]}
    report = {"schema": "issue215-baseline-causal-comparison-1", "evidence": {"baseline": {"path": str(args.baseline), "sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest()}, "candidate": {"path": str(args.candidate), "sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest()}}, "scope": candidate["scope"], "trimming": "OFF", "profile": candidate["profile"], "selected_runtime": candidate["runtime"], "artifact": candidate["nemotron_artifact"], "execution_source_sha256": candidate.get("execution_source_sha256"), "gpu_asr": candidate["gpu_asr"], "cases": []}
    for case in candidate["cases"]:
        name = case["case"]
        def vad_schedule(value):
            return [
                (item["genuine_onset"], item["seal_reason"], ranges(item["content_ranges"]), ranges(item["context_ranges"]))
                for item in value.get("segments", [])
            ]
        report["cases"].append({
            "case": name,
            "same_input_pcm": old[name]["input_pcm_f32_sha256"] == case["input_pcm_f32_sha256"],
            "same_asr_options": old[name].get("asr") == case.get("asr"),
            "same_capture_vad_range_schedule": vad_schedule(old[name]) == vad_schedule(case),
            "baseline": summarize(old[name]),
            "new": summarize(case),
        })
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
