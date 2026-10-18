from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from issue218_policy_summary import analyze
from issue264_validation_prerequisites import ROOT, sanitize


def native_failures(case):
    groups = []
    linked = {}
    for index, event in enumerate(case["trace"]):
        boundary = event["event"]
        value = event.get("value", {})
        fields = value
        if boundary in ("native_result", "provider_result"):
            outcome = value.get("outcome")
            if outcome not in ("failed", "expired", "degraded"):
                continue
            reason = value.get("failure_reason")
        elif boundary == "gpu_asr_diagnostic":
            kind = value.get("kind")
            fields = value.get("fields", {})
            result = fields.get("result")
            if kind == "decode_attempt" and result and result != "success":
                outcome = "expired" if result == "expired" else "failed"
                reason = result
            elif kind in ("work_prestart_failed", "work_expired"):
                outcome = "expired" if kind == "work_expired" else "failed"
                reason = fields.get("failure_reason", fields.get("failure", fields.get("reason")))
            else:
                continue
        else:
            continue
        identity = None
        if isinstance(fields.get("identity"), dict) and fields["identity"].get("provider_turn_id"):
            identity = {"kind": "identity", "value": fields["identity"]}
        else:
            explicit = {
                key: fields[key]
                for key in ("request_id", "attempt_id")
                if fields.get(key) not in (None, "")
            }
            if explicit:
                identity = {"kind": "explicit_request_attempt", "value": explicit}
        key = (
            json.dumps([identity, outcome, reason], sort_keys=True)
            if identity is not None
            else None
        )
        occurrence = {"trace_index": index, "boundary": boundary, "record": event}
        if key is not None and key in linked:
            groups[linked[key]]["occurrences"].append(occurrence)
        else:
            if key is not None:
                linked[key] = len(groups)
            groups.append(
                {
                    "outcome": outcome,
                    "failure_reason": reason,
                    "explicit_identity": identity,
                    "occurrences": [occurrence],
                }
            )
    return groups


def assignment_key(assignment):
    attribution = assignment.get("attribution") or {}
    key = attribution.get("key")
    if not key:
        return None
    return key


def late_assignments(case):
    observations = [
        event for event in case["trace"] if event["event"] == "presentation_assignments"
    ]
    result = []
    for item in case["sources"]:
        source = item["source"]
        source_id = source["source_id"]
        records = [
            {"observed_at": event["at"], **event["records"][source_id]}
            for event in observations
            if source_id in event["records"]
        ]
        readable = next(
            (
                record["first_readable_at"]
                for record in records
                if record["first_readable_at"] is not None
            ),
            None,
        )
        allocated = [
            record
            for record in records
            if record["assignment"]["settled"]
            and assignment_key(record["assignment"]) is not None
            and (
                record["assignment"].get("palette_index") is not None
                or record["assignment"].get("palette_overflow")
            )
        ]
        predisplay = next(
            (record for record in allocated if record["first_readable_at"] is None), None
        )
        first_allocation = predisplay or next(iter(allocated), None)
        submission = next(
            (
                event["submission"]
                for event in case["trace"]
                if event["event"] == "ordered_publication_wait"
                and event["submission"]["source_child"]["source_id"] == source_id
            ),
            None,
        )
        logical_key = None if submission is None else "peer:" + submission["child_utterance_id"]
        first_block = next(
            (
                block
                for snapshot in case["renderer_snapshots"]
                for block in snapshot["snapshot"]["blocks"]
                if logical_key is not None and block.get("logical_turn_key") == logical_key
            ),
            None,
        )
        initially_colored = first_block is not None and first_block.get("speaker_style") in (
            "gold",
            "cyan",
            "coral",
            "blue",
        )
        upper = None if first_allocation is None else first_allocation["observed_at"]
        delay = None if upper is None or readable is None else max(0, upper - readable)
        if predisplay is not None or initially_colored:
            band = "identified_before_first_readable"
        elif readable is None:
            band = "not_observed_readable"
        elif delay is None:
            band = "unresolved"
        elif delay <= 0.5:
            band = "late_assignment_observed_within_500ms"
        elif delay <= 1.0:
            band = "late_assignment_observed_within_1s"
        else:
            band = "assignment_observation_after_1s_requires_raw_deadline_review"
        expected_inputs = set()
        for origin in source["origins"]:
            for span in origin.get("source_ranges", []):
                left, right = span.get("normalized_start_sample"), span.get("normalized_end_sample")
                if left is None or right is None:
                    continue
                for reference in case["assembly_scoring_only"]:
                    begin, end = reference["scenario_sample_range"]
                    if left < end and begin < right:
                        expected_inputs.add(reference["input"])
        result.append(
            {
                "source_id": source_id,
                "source_text": source["raw_text"],
                "immutable_source_speaker": source["speaker"],
                "first_readable_block": first_block,
                "immutable_origin_attribution": [
                    origin.get("speaker_attribution") for origin in source["origins"]
                ],
                "first_readable_monotonic": readable,
                "first_allocated_assignment_observed_upper_bound": upper,
                "late_delay_observation_upper_bound_seconds": delay,
                "band": band,
                "assignment": None if first_allocation is None else first_allocation["assignment"],
                "last_authority_record": records[-1] if records else None,
                "scoring_reference_inputs_overlapping_original_source_ranges": sorted(
                    expected_inputs
                ),
                "scoring_limit": "Assembly footprint/reference only, never fed to product decisions. Mixed footprints include crop quiet edges and are not automatically wrong mappings. Correct A/B/return identity is judged against these coordinates and actual evidence, not identity count or text similarity.",
            }
        )
    return result


def presentation_transitions(case):
    previous = None
    transitions = []
    conserved_fields = (
        "id",
        "occupant_key",
        "appearance_seq",
        "channel",
        "block_variant",
        "primary_text",
        "secondary_text",
        "secondary_enabled",
        "primary_language",
        "secondary_language",
        "logical_turn_key",
        "publication_scope",
        "publication_generation",
        "publication_order",
        "publication_index",
    )
    for snapshot in case["renderer_snapshots"]:
        if previous is not None:
            old_blocks = previous["snapshot"]["blocks"]
            old_by_id = {block["id"]: block for block in old_blocks}
            blocks = snapshot["snapshot"]["blocks"]
            for block in blocks:
                old = old_by_id.get(block["id"])
                if old is None or old.get("speaker_style") == block.get("speaker_style"):
                    continue
                changed = {
                    field: [old.get(field), block.get(field)]
                    for field in conserved_fields
                    if old.get(field) != block.get(field)
                }
                transitions.append(
                    {
                        "card_id": block["id"],
                        "from_style": old.get("speaker_style"),
                        "to_style": block.get("speaker_style"),
                        "submitted_at": snapshot["submitted_at"],
                        "renderer_ack_at": snapshot.get("software_commit_observed_at"),
                        "changed_conservation_fields": changed,
                        "card_order_unchanged": [item["id"] for item in old_blocks]
                        == [item["id"] for item in blocks],
                        "expiration_before": (previous.get("block_expirations") or {}).get(
                            block["id"]
                        ),
                        "expiration_after": (snapshot.get("block_expirations") or {}).get(
                            block["id"]
                        ),
                        "revision": snapshot["snapshot"]["revision"],
                        "limit": "A simultaneous source/translation completion may lawfully change text; controlled metadata-only caption cases separately protect pure-style conservation.",
                    }
                )
        previous = snapshot
    return transitions


def summarize(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    result = analyze(path)
    for case, row in zip(report["cases"], result["cases"]):
        row["pipeline_status_is_not_quality_pass"] = True
        row["native_failures"] = native_failures(case)
        row["observation_errors"] = [
            event for event in case["trace"] if event["event"] == "observation_exception"
        ]
        assignments = late_assignments(case)
        row["presentation_assignments"] = assignments
        row["presentation_resolution_bands"] = dict(Counter(item["band"] for item in assignments))
        row["presentation_style_transitions"] = presentation_transitions(case)
        row["authority_resolution_counters"] = case.get("presentation_resolution_counts")
        row["enclosing_completion_observations"] = [
            {"at": event["at"], "owned_event": event["value"]}
            for event in case["trace"]
            if event["event"] == "vad" and event["value"]["event"].get("reason")
        ]
        row["retained_evidence_peaks"] = {
            "frames": max(
                (
                    event["retained_frames"]
                    for event in case["trace"]
                    if event["event"] == "presentation_assignments"
                ),
                default=0,
            ),
            "proofs": max(
                (
                    event["retained_proofs"]
                    for event in case["trace"]
                    if event["event"] == "presentation_assignments"
                ),
                default=0,
            ),
        }
    result["schema"] = "issue264-validation-summary-1"
    result["recorded_pipeline_status"] = result.pop("status")
    result["scenario_disposition"] = (
        "invalid_observation"
        if any(row["observation_errors"] for row in result["cases"])
        else (
            "failed_native_execution"
            if any(row["native_failures"] for row in result["cases"])
            else "not_automatically_assessed"
        )
    )
    result["frozen_source_analysis_limit"] = (
        "Inherited attribution-state/count/clear-AB/return fields describe immutable source-time metadata only, not final presentation resolution; inspect presentation_assignments and actual source-keyed caption observations for #264."
    )
    result["raw_trace_sha256"] = hashlib.sha256(raw).hexdigest()
    result["interpretation"] = (
        "Changed-path smoke only: one repetition, no acoustic pass-rate threshold, no profile selection/statistics. Native replay is historical ASR with actual current downstream inference/rendering. Source immutable snapshots and presentation assignment are separate. Pixel validity requires actual image observation; ACK is not visibility."
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != ROOT / "native/nemotron" or not output.name.startswith(
        "issue264_validation_"
    ):
        raise ValueError("Only assigned task-local reports are permitted")
    if output.exists():
        raise FileExistsError("Preserve previous summary attempts")
    results = [summarize(path.resolve()) for path in args.trace]
    output.write_text(
        sanitize({"schema": "issue264-validation-summary-bundle-1", "runs": results}),
        encoding="utf-8",
    )
    print(
        sanitize(
            {
                "runs": [
                    {
                        "candidate": result["candidate"],
                        "route": result["route"],
                        "cases": len(result["cases"]),
                        "recorded_pipeline_status": result["recorded_pipeline_status"],
                        "scenario_disposition": result["scenario_disposition"],
                    }
                    for result in results
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
