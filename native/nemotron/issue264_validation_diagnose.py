from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

from issue264_validation_prerequisites import ROOT, sanitize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != ROOT / "native/nemotron" or not output.name.startswith(
        "issue264_validation_"
    ):
        raise ValueError("Only assigned task-local reports are permitted")
    if output.exists():
        raise FileExistsError("Preserve prior diagnosis attempts")
    trace_path = args.trace.resolve()
    raw = trace_path.read_bytes()
    report = json.loads(raw)
    case = report["cases"][0]
    archive_path = trace_path.parent / "execution_sources.zip"
    with zipfile.ZipFile(archive_path) as archive:
        checked = {}
        for relative in (
            "src/puripuly_heart/core/speaker/evidence.py",
            "src/puripuly_heart/core/stt/speaker_attribution.py",
            "src/puripuly_heart/core/stt/batch_speaker.py",
        ):
            actual = hashlib.sha256(archive.read(relative)).hexdigest()
            if actual != report["sources_sha256"][relative]:
                raise ValueError("Frozen execution source hash mismatch")
            checked[relative] = actual

    def audit(event, _args):
        if event in ("socket.connect", "socket.bind", "subprocess.Popen"):
            raise PermissionError("Offline diagnosis prohibits processes and network")

    sys.addaudithook(audit)
    sys.path.insert(0, str(archive_path) + "/src")
    from puripuly_heart.core.audio.format import AudioCaptureSpan
    from puripuly_heart.core.speaker.contracts import (
        ProbabilityFrame,
        SourceMapPiece,
        SpeakerEvidence,
    )
    from puripuly_heart.core.speaker.evidence import SpeakerEvidenceView
    from puripuly_heart.core.stt.speaker_attribution import attribute_speaker, partition_replay

    def span(value):
        return AudioCaptureSpan(**value)

    def evidence(value):
        fields = dict(value)
        fields["source_map"] = tuple(
            SourceMapPiece(
                **{**piece, "capture": None if piece["capture"] is None else span(piece["capture"])}
            )
            for piece in value["source_map"]
        )
        frame = value["frame"]
        fields["frame"] = (
            None
            if frame is None
            else ProbabilityFrame(
                **{
                    **frame,
                    "probabilities": tuple(frame["probabilities"]),
                    "active_slots": tuple(frame["active_slots"]),
                }
            )
        )
        return SpeakerEvidence(**fields)

    def frame_record(frame):
        return {
            "capture_range": [frame.span.normalized_start_sample, frame.span.normalized_end_sample],
            "compact_end": frame.compact_end,
            "ready_at": frame.ready_at,
            "closed": frame.closed,
            "epoch": frame.epoch,
            "slots": frame.slots,
            "clear": frame.clear,
            "uncertain": frame.uncertain,
            "competing": frame.competing,
            "probabilities": frame.probabilities,
        }

    def diagnose(primary, replayed, view):
        captured = {}

        def observe_return(frame, event, _arg):
            if event == "return" and frame.f_code is attribute_speaker.__code__:
                captured.update(frame.f_locals)
            return observe_return

        previous_trace = sys.gettrace()
        sys.settrace(observe_return)
        try:
            result = attribute_speaker(primary, view.frames, view.immutable, replayed=replayed)
        finally:
            sys.settrace(previous_trace)
        scalar_names = (
            "missing",
            "broken",
            "overlap",
            "candidate",
            "clear_slots",
            "active_slots",
            "support",
            "total",
            "longest",
            "exceeds",
            "foreign_replay",
            "foreign_uncertainty",
            "epochs",
            "state",
        )
        details = {name: captured.get(name) for name in scalar_names}
        details = {
            name: sorted(value) if isinstance(value, set) else value
            for name, value in details.items()
        }
        candidate = captured.get("candidate")
        ambiguous = []
        foreign = []
        for capture, left, right, rate, frame, is_replay in captured.get("pieces", []):
            if frame.uncertain or frame.competing or len(frame.slots) > 1:
                item = {
                    "range": [left, right],
                    "seconds": (right - left) / rate,
                    "is_replay": is_replay,
                    "frame": frame_record(frame),
                }
                ambiguous.append(item)
                if (
                    candidate is not None
                    and not is_replay
                    and frame.uncertain
                    and frame.probabilities
                ):
                    peak = max(frame.probabilities)
                    if peak >= 0.35 and frame.probabilities[candidate] < peak:
                        foreign.append(item)
        return {
            "result": asdict(result),
            "classifier_locals": details,
            "uncertain_or_overlap_pieces": ambiguous,
            "foreign_leaning_unique_pieces": foreign,
            "accepted_frontiers": {
                "processed": view.processed,
                "immutable": view.immutable,
                "revision": view.revision,
                "generation": view.generation,
                "frames": len(view.frames),
            },
        }

    view = SpeakerEvidenceView()
    pending = iter(case["evidence"])
    consumed = 0
    submitted = []
    requests = {}
    proof_requests = {}
    cuts = []
    own_finals = []
    readable = {}
    settlements = {}
    raw_sources = {item["source"]["source_id"]: item["source"] for item in case["sources"]}
    for event in case["trace"]:
        kind = event["event"]
        if kind == "evidence_poll":
            batch = [evidence(next(pending)) for _ in range(event["count"])]
            consumed += len(batch)
            if batch:
                view.observe(batch)
        elif kind == "native_submit":
            source = tuple(span(value) for value in event["sealed_content_ranges"])
            context = tuple(
                span(value)
                for payload in event["payloads"]
                if payload["context_only"]
                for value in payload["source_ranges"]
            )
            unique_context, replayed = partition_replay(context, submitted)
            request_id = event["identity"]["provider_turn_id"]
            requests[request_id] = (source + unique_context, replayed)
            start = min(value.normalized_start_sample for value in source)
            end = max(value.normalized_end_sample for value in source)
            transitions = []
            for boundary, predecessor, successor in view.transitions():
                if abs(boundary.normalized_sample - end) > 6400:
                    continue
                successor_end = successor.span.normalized_end_sample
                confirming = next(
                    (
                        frame
                        for frame in view.stable_frames()
                        if frame.span.normalized_start_sample == successor_end
                        and frame.clear
                        and frame.slots == successor.slots
                    ),
                    None,
                )
                transitions.append(
                    {
                        "boundary": asdict(boundary),
                        "predecessor": frame_record(predecessor),
                        "successor_first_clear": frame_record(successor),
                        "successor_confirming_temporal_frame": (
                            None if confirming is None else frame_record(confirming)
                        ),
                        "active_successor_onset_frames": [
                            frame_record(frame)
                            for frame in view.frames
                            if frame.span.normalized_start_sample >= end - 320
                            and frame.span.normalized_end_sample <= boundary.normalized_sample
                            and frame.slots == successor.slots
                        ],
                    }
                )
            cuts.append(
                {
                    "at": event["at"],
                    "request_id": request_id,
                    "cut_reason": event["seal_reason"],
                    "source_range_bounds": [start, end],
                    "physical_cursor_before": max(
                        (value.normalized_end_sample for value in submitted), default=-1
                    ),
                    "physical_cursor_after": end,
                    "submitted_pcm_samples": sum(
                        payload["samples"] for payload in event["payloads"]
                    ),
                    "request_pcm_sha256": event["pcm16_sha256"],
                    "all_payload_pcm_match_capture": all(
                        payload["matches_declared_native_input"] for payload in event["payloads"]
                    ),
                    "source_ranges": [asdict(value) for value in source],
                    "unique_context_ranges": [asdict(value) for value in unique_context],
                    "proven_replay_ranges": [asdict(value) for value in replayed],
                    "confirmed_transitions_near_cut": transitions,
                }
            )
            submitted.extend((*context, *source))
        elif kind == "own_native_final_attribution":
            request_id = event["identity"]["provider_turn_id"]
            proof_requests[event["proof_id"]] = request_id
            diagnostic = diagnose(*requests[request_id], view)
            own_finals.append(
                {
                    "at": event["at"],
                    "request_id": request_id,
                    "proof_id": event["proof_id"],
                    "recorded_result": event["result"],
                    "diagnostic": diagnostic,
                    "exact_recorded_frontiers_match": view.revision == event["accepted_revision"]
                    and view.immutable == event["immutable"]
                    and view.processed == event["processed"]
                    and len(view.frames) == event["delivered_frame_count"],
                    "reconstructed_attribution_matches": diagnostic["result"]["attribution_state"]
                    == event["result"]["attribution_state"],
                }
            )
        elif kind == "presentation_assignments":
            for source_id, snapshot in event["records"].items():
                source = raw_sources[source_id]
                proofs = [
                    origin.get("speaker_proof_id")
                    for origin in source["origins"]
                    if origin.get("speaker_proof_id")
                ]
                request_ids = list(dict.fromkeys(proof_requests[proof] for proof in proofs))
                if snapshot["first_readable_at"] is not None and source_id not in readable:
                    readable[source_id] = {
                        "observed_at": event["at"],
                        "source_text": source["raw_text"],
                        "snapshot": snapshot,
                        "proof_ids": proofs,
                        "diagnostics": [
                            diagnose(*requests[request], view) for request in request_ids
                        ],
                    }
                if (
                    snapshot["first_readable_at"] is not None
                    and snapshot["assignment"]["settled"]
                    and source_id not in settlements
                ):
                    settlements[source_id] = {
                        "observed_at": event["at"],
                        "source_text": source["raw_text"],
                        "snapshot": snapshot,
                        "proof_ids": proofs,
                        "diagnostics": [
                            diagnose(*requests[request], view) for request in request_ids
                        ],
                    }
    if consumed != len(case["evidence"]):
        raise ValueError("Did not reconstruct every delivered evidence item")
    failures = [
        event
        for event in case["trace"]
        if event["event"] == "gpu_asr_diagnostic"
        and (
            event["value"]["kind"].startswith("worker_recovery")
            or (
                event["value"]["kind"] == "decode_attempt"
                and event["value"]["fields"].get("result") == "decode_failure"
            )
        )
    ]
    diagnosis = {
        "schema": "issue264-frozen-first04-diagnosis-1",
        "candidate": report["candidate"],
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "runtime_driver_sha256": report["driver_sha256"],
        "frozen_source_hashes": checked,
        "loaded_frozen_modules": {
            name: module.__file__
            for name, module in tuple(sys.modules.items())
            if name.startswith("puripuly_heart") and getattr(module, "__file__", None)
        },
        "scope": "Deterministic reconstruction of original accepted probability deliveries through exact archived candidate code. No ASR, speaker model, translation, GUI, process or network execution. No acoustic rerun; no parameter changes.",
        "cut_requests": cuts,
        "own_native_finals": own_finals,
        "first_readable_source_observations": readable,
        "terminal_assignment_observations": settlements,
        "decode_failures_and_recovery": failures,
        "all_own_final_frontiers_and_states_reproduced": all(
            item["exact_recorded_frontiers_match"] and item["reconstructed_attribution_matches"]
            for item in own_finals
        ),
        "observation_limit": "First-readable and deadline snapshots are the first recorded public source snapshots, with observation upper bounds. No unrecorded callback timestamp or physical display onset is manufactured. Proof ranges reconstructed from original native PCM/source maps and proven replay partition; no arrival-order/text correspondence guess.",
    }
    output.write_text(sanitize(diagnosis), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": output.name,
                "cuts": len(cuts),
                "own_finals": len(own_finals),
                "all_own_final_frontiers_and_states_reproduced": diagnosis[
                    "all_own_final_frontiers_and_states_reproduced"
                ],
                "foreign_leaning_unique_ranges": [
                    {
                        "request_id": item["request_id"],
                        "ranges": [
                            piece["range"]
                            for piece in item["diagnostic"]["foreign_leaning_unique_pieces"]
                        ],
                        "classifier": item["diagnostic"]["classifier_locals"],
                    }
                    for item in own_finals
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
