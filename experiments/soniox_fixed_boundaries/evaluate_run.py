from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

RATE = 16_000
WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def normalized_words(text: str) -> list[str]:
    return [item.casefold() for item in WORD_RE.findall(text)]


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int, int]:
    rows: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0)] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)
    ]
    for i in range(1, len(reference) + 1):
        rows[i][0] = (i, 0, i, 0)
    for j in range(1, len(hypothesis) + 1):
        rows[0][j] = (j, 0, 0, j)
    for i, expected in enumerate(reference, 1):
        for j, actual in enumerate(hypothesis, 1):
            if expected == actual:
                rows[i][j] = rows[i - 1][j - 1]
                continue
            rows[i][j] = min(
                tuple(a + b for a, b in zip(rows[i - 1][j - 1], (1, 1, 0, 0), strict=True)),
                tuple(a + b for a, b in zip(rows[i - 1][j], (1, 0, 1, 0), strict=True)),
                tuple(a + b for a, b in zip(rows[i][j - 1], (1, 0, 0, 1), strict=True)),
            )
    return rows[-1][-1]


def provider_to_source(
    plan: dict[str, Any], sample: int
) -> tuple[int | None, str | None, str | None]:
    for event in plan["events"]:
        if event.get("event") != "audio":
            continue
        if event["provider_start_sample"] <= sample < event["provider_end_sample"]:
            source = event.get("source_start_sample")
            return (
                None if source is None else source + sample - event["provider_start_sample"],
                event.get("segment_id"),
                event.get("kind"),
            )
    return None, None, None


def load_trace(path: Path, plan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    facts: dict[str, Any] = {"errors": [], "gate_wait_ms": [], "max_backlog_ms": 0.0}
    seen: set[tuple[Any, ...]] = set()
    start_ns = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        kind = event["event"]
        if kind == "session_start":
            start_ns = event["monotonic_ns"]
        elif kind == "audio_sent":
            facts["max_backlog_ms"] = max(facts["max_backlog_ms"], event["actual_backlog_ms"])
        elif kind == "finalize_sent":
            facts.setdefault("segments", {}).setdefault(event["segment_id"], {}).update(
                {
                    "source_seal_sample": event["source_seal_sample"],
                    "speech_end_source_sample": event.get("speech_end_source_sample"),
                    "finalize_sent_offset_ms": (event["send_finished_monotonic_ns"] - start_ns) / 1e6,
                }
            )
        elif kind == "finalize_gate_released":
            segment = facts.setdefault("segments", {}).setdefault(event["segment_id"], {})
            segment["primary_ready_offset_ms"] = (event["receipt_monotonic_ns"] - start_ns) / 1e6
            segment["gate_wait_ms"] = event["wait_ms"]
            facts["gate_wait_ms"].append(event["wait_ms"])
        elif kind == "connection_error":
            facts["errors"].append({"type": event["error_type"], "message": event["error"]})
        elif kind != "provider_receipt":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        for field in ("final_audio_proc_ms", "total_audio_proc_ms"):
            if field in payload:
                facts[field] = payload[field]
        for token in payload.get("tokens", []):
            if not isinstance(token, dict) or token.get("is_final") is not True or token.get("text") == "<fin>":
                continue
            key = (token.get("start_ms"), token.get("end_ms"), token.get("text"), token.get("speaker"))
            if key in seen:
                continue
            seen.add(key)
            center = round((float(token.get("start_ms", 0)) + float(token.get("end_ms", 0))) * RATE / 2000)
            source_sample, segment_id, source_kind = provider_to_source(plan, center)
            tokens.append(
                {
                    "text": str(token.get("text", "")),
                    "speaker": str(token.get("speaker", "unknown")),
                    "source_sample": source_sample,
                    "planned_segment_id": segment_id,
                    "receipt_segment_id": event.get("attributed_segment_id"),
                    "receipt_offset_ms": (event["monotonic_ns"] - start_ns) / 1e6,
                    "source_kind": source_kind,
                }
            )
    for token in tokens:
        segment_id = token["receipt_segment_id"]
        if segment_id is None:
            continue
        segment = facts.setdefault("segments", {}).setdefault(segment_id, {})
        first = segment.get("first_speaker_token_offset_ms")
        if first is None or token["receipt_offset_ms"] < first:
            segment["first_speaker_token_offset_ms"] = token["receipt_offset_ms"]
    for segment in facts.get("segments", {}).values():
        seal = segment.get("source_seal_sample")
        speech_end = segment.get("speech_end_source_sample")
        sent = segment.get("finalize_sent_offset_ms")
        ready = segment.get("primary_ready_offset_ms")
        first = segment.get("first_speaker_token_offset_ms")
        if seal is not None and sent is not None:
            segment["seal_to_finalize_ms"] = sent - seal * 1000 / RATE
        if speech_end is not None and ready is not None:
            segment["speech_end_to_primary_ready_ms"] = ready - speech_end * 1000 / RATE
        if speech_end is not None and first is not None:
            segment["speech_end_to_first_speaker_label_ms"] = first - speech_end * 1000 / RATE
    return tokens, facts


def reference_words(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = [item for item in payload["items"] if item["kind"] == "word"]
    return items, sorted({str(item["speaker_id"]) for item in items})


def reference_at(items: list[dict[str, Any]], sample: int) -> list[dict[str, Any]]:
    return [
        item
        for item in items
        if item["source_start_sample"] <= sample < max(item["source_end_sample"], item["source_start_sample"] + 1)
    ]


def fixed_mapping(tokens: list[dict[str, Any]], reference: list[dict[str, Any]], speakers: list[str]) -> dict[str, str]:
    provider_speakers = sorted({token["speaker"] for token in tokens if token["speaker"] != "unknown"})
    scores: defaultdict[tuple[str, str], int] = defaultdict(int)
    for token in tokens:
        if token["source_sample"] is None:
            continue
        for item in reference_at(reference, token["source_sample"]):
            scores[(token["speaker"], item["speaker_id"])] += 1
    targets = speakers + [f"unmapped-{index}" for index in range(max(0, len(provider_speakers) - len(speakers)))]
    best: tuple[int, dict[str, str]] = (-1, {})
    for assigned in itertools.permutations(targets, len(provider_speakers)):
        mapping = dict(zip(provider_speakers, assigned, strict=True))
        candidate = sum(scores[(source, target)] for source, target in mapping.items())
        if candidate > best[0]:
            best = candidate, mapping
    return best[1]


def evaluate_stream(tokens: list[dict[str, Any]], facts: dict[str, Any], reference: list[dict[str, Any]], speakers: list[str], plan: dict[str, Any]) -> dict[str, Any]:
    source_mapped = [token for token in tokens if token["source_sample"] is not None]
    mapped = sorted(
        (
            token
            for token in source_mapped
            if token["source_kind"] in {"real_content", "continuous_observation"}
        ),
        key=lambda token: token["source_sample"],
    )
    source_intervals = [
        (event["source_start_sample"], event["source_end_sample"])
        for event in plan["events"]
        if event.get("event") == "audio"
        and event.get("source_start_sample") is not None
        and event.get("kind") != "prefix_context"
    ]
    reference_scope = [
        item
        for item in reference
        if any(
            left <= (item["source_start_sample"] + item["source_end_sample"]) // 2 < right
            for left, right in source_intervals
        )
    ]
    availability_ms = sorted(
        token["receipt_offset_ms"] - token["source_sample"] * 1000 / RATE for token in mapped
    )

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        return values[round((len(values) - 1) * fraction)]
    ref = [part for item in reference_scope for part in normalized_words(str(item["text"]))]
    hyp = [part for token in mapped for part in normalized_words(token["text"])]
    distance, substitutions, deletions, insertions = edit_counts(ref, hyp)
    mapping = fixed_mapping(mapped, reference, speakers)
    speaker_total = speaker_correct = overlap_total = overlap_correct = 0
    concrete: list[dict[str, Any]] = []
    for token in mapped:
        matches = reference_at(reference, token["source_sample"])
        if not matches:
            if len(concrete) < 20:
                concrete.append({"kind": "unaligned_provider_token", "source_sample": token["source_sample"], "segment_id": token["planned_segment_id"]})
            continue
        correct = any(item["speaker_id"] == mapping.get(token["speaker"]) for item in matches)
        speaker_total += 1
        speaker_correct += int(correct)
        if len({item["speaker_id"] for item in matches}) > 1:
            overlap_total += 1
            overlap_correct += int(correct)
        if not correct and len(concrete) < 20:
            concrete.append({"kind": "speaker_mismatch", "source_sample": token["source_sample"], "segment_id": token["planned_segment_id"], "reference_speakers": sorted({item["speaker_id"] for item in matches})})
    segment_metrics: list[dict[str, Any]] = []
    segment_order = [
        event["segment_id"] for event in plan["events"] if event.get("event") == "finalize"
    ]
    for index, segment_id in enumerate(segment_order):
        segment_events = [
            event
            for event in plan["events"]
            if event.get("event") == "audio"
            and event.get("kind") == "real_content"
            and event.get("segment_id") == segment_id
        ]
        if not segment_events:
            continue
        left = min(event["source_start_sample"] for event in segment_events)
        right = max(event["source_end_sample"] for event in segment_events)
        scoped_reference = [
            item
            for item in reference
            if left <= (item["source_start_sample"] + item["source_end_sample"]) // 2 < right
        ]
        scoped_tokens = [token for token in mapped if token["planned_segment_id"] == segment_id]
        expected = [
            part for item in scoped_reference for part in normalized_words(str(item["text"]))
        ]
        actual = [part for token in scoped_tokens for part in normalized_words(token["text"])]
        seg_distance, seg_subs, seg_deletions, seg_insertions = edit_counts(expected, actual)
        segment_metrics.append(
            {
                "segment_id": segment_id,
                "source_span": [left, right],
                "following_segment_id": (
                    segment_order[index + 1] if index + 1 < len(segment_order) else None
                ),
                "reference_words": len(expected),
                "hypothesis_words": len(actual),
                "distance": seg_distance,
                "substitutions": seg_subs,
                "deletions": seg_deletions,
                "insertions": seg_insertions,
                "wer": seg_distance / len(expected) if expected else None,
            }
        )
    leakage = sum(
        token["planned_segment_id"] != token["receipt_segment_id"]
        for token in mapped
        if token["receipt_segment_id"] is not None
    )
    return {
        "final_token_count": len(tokens),
        "source_mapped_token_count": len(source_mapped),
        "fixed_speaker_mapping": mapping,
        "reference_word_count": len(ref),
        "hypothesis_word_count": len(hyp),
        "word_error": {
            "distance": distance,
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "wer": distance / len(ref) if ref else None,
        },
        "speaker": {
            "correct": speaker_correct,
            "total": speaker_total,
            "accuracy": speaker_correct / speaker_total if speaker_total else None,
            "overlap_correct": overlap_correct,
            "overlap_total": overlap_total,
        },
        "segment_text_metrics": segment_metrics,
        "alignment_coverage": {
            "reference_words_in_planned_scope": len(reference_scope),
            "provider_tokens_source_mapped": len(source_mapped),
            "provider_tokens_scored_content": len(mapped),
            "provider_prefix_context_tokens": sum(
                token["source_kind"] == "prefix_context" for token in source_mapped
            ),
            "provider_tokens_unmapped": len(tokens) - len(source_mapped),
            "speaker_scored_tokens": speaker_total,
            "overlap_scored_tokens": overlap_total,
            "sequential_scored_tokens": speaker_total - overlap_total,
            "token_availability_ms": {
                "count": len(availability_ms),
                "p50": percentile(availability_ms, 0.50),
                "p95": percentile(availability_ms, 0.95),
                "max": max(availability_ms) if availability_ms else None,
            },
        },
        "cross_boundary_receipt_token_count": leakage,
        "timing": facts,
        "concrete_source_failures": concrete,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one Issue #157 Soniox run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "manifest.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    plan_payload = json.loads((args.run_dir / "plan.json").read_text(encoding="utf-8"))
    by_recording = {recording["id"]: recording for recording in manifest["recordings"]}
    results = []
    missing_traces: list[str] = []
    for plan in plan_payload["plans"]:
        recording = by_recording[plan["recording_id"]]
        reference, speakers = reference_words(args.manifest.parent / recording["human_reference"]["path"])
        trace_path = args.run_dir / f"{plan['recording_id']}--{plan['arm']}--{plan['stream_role']}.jsonl"
        if not trace_path.is_file():
            missing_traces.append(trace_path.name)
            continue
        tokens, facts = load_trace(trace_path, plan)
        results.append(
            {
                "recording_id": plan["recording_id"],
                "arm": plan["arm"],
                "stream_role": plan["stream_role"],
                "metrics": evaluate_stream(tokens, facts, reference, speakers, plan),
            }
        )
    output = {
        "status": "evaluated",
        "execution_identity": plan_payload["execution_identity"],
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coverage": plan_payload["coverage"],
        "streams": results,
        "missing_traces": missing_traces,
    }
    text = json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
