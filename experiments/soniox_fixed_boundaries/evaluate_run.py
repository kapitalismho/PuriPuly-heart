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
def concatenated_piece_runs(
    tokens: list[dict[str, Any]], include: Any
) -> list[str]:
    words: list[str] = []
    run: list[str] = []
    for token in tokens:
        if include(token):
            run.append(str(token["text"]))
            continue
        if run:
            words.extend(normalized_words("".join(run)))
            run.clear()
    if run:
        words.extend(normalized_words("".join(run)))
    return words




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
    plan: dict[str, Any], sample: int, receipt_segment_id: str | None
) -> tuple[int | None, str | None, str | None]:
    if receipt_segment_id is None:
        return None, None, "missing_receipt_ownership"
    owned_audio = [
        event
        for event in plan["events"]
        if event.get("event") == "audio" and event.get("segment_id") == receipt_segment_id
    ]
    for event in owned_audio:
        if event["provider_start_sample"] <= sample < event["provider_end_sample"]:
            source = event.get("source_start_sample")
            return (
                None if source is None else source + sample - event["provider_start_sample"],
                receipt_segment_id,
                event.get("kind"),
            )
    return None, receipt_segment_id, "timestamp_outside_receipt_scope"
def provider_to_source_unscoped(
    plan: dict[str, Any], sample: int
) -> tuple[int | None, str | None]:
    for event in plan["events"]:
        if (
            event.get("event") == "audio"
            and event["provider_start_sample"] <= sample < event["provider_end_sample"]
        ):
            source = event.get("source_start_sample")
            return (
                None if source is None else source + sample - event["provider_start_sample"],
                event.get("kind"),
            )
    return None, None




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
            center = round(
                (float(token.get("start_ms", 0)) + float(token.get("end_ms", 0)))
                * RATE
                / 2000
            )
            receipt_segment_id = event.get("attributed_segment_id")
            source_sample, segment_id, source_kind = provider_to_source(
                plan, center, receipt_segment_id
            )
            unscoped_source_sample, unscoped_source_kind = provider_to_source_unscoped(
                plan, center
            )
            tokens.append(
                {
                    "text": str(token.get("text", "")),
                    "speaker": str(token.get("speaker", "unknown")),
                    "source_sample": source_sample,
                    "planned_segment_id": segment_id,
                    "receipt_segment_id": receipt_segment_id,
                    "receipt_offset_ms": (event["monotonic_ns"] - start_ns) / 1e6,
                    "source_kind": source_kind,
                    "unscoped_source_sample": unscoped_source_sample,
                    "unscoped_source_kind": unscoped_source_kind,
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

def reference_words(
    path: Path,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = [item for item in payload["items"] if item["kind"] == "word"]
    turns = list(payload.get("turns", []))
    return items, sorted({str(item["speaker_id"]) for item in items}), turns


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


def evaluate_stream(tokens: list[dict[str, Any]], facts: dict[str, Any], reference: list[dict[str, Any]], speakers: list[str], turns: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
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
    unscoped_timestamp_tokens = sorted(
        [
            token
            for token in tokens
            if token["unscoped_source_sample"] is not None
            and token["unscoped_source_kind"] in {"real_content", "continuous_observation"}
        ],
        key=lambda token: token["unscoped_source_sample"],
    )
    unscoped_timestamp_hypothesis = normalized_words(
        "".join(str(token["text"]) for token in unscoped_timestamp_tokens)
    )
    unscoped_timestamp_edits = edit_counts(ref, unscoped_timestamp_hypothesis)
    hyp = normalized_words("".join(str(token["text"]) for token in mapped))
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
    finalize_by_segment = {
        event["segment_id"]: event
        for event in plan["events"]
        if event.get("event") == "finalize"
    }
    for index, segment_id in enumerate(segment_order):
        segment_events = [
            event
            for event in plan["events"]
            if event.get("event") == "audio"
            and event.get("kind") in {"real_content", "continuous_observation"}
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
        following_id = segment_order[index + 1] if index + 1 < len(segment_order) else None
        following_events = [
            event
            for event in plan["events"]
            if event.get("event") == "audio"
            and event.get("kind") == "real_content"
            and event.get("segment_id") == following_id
        ]
        following_left = (
            min(event["source_start_sample"] for event in following_events)
            if following_events
            else None
        )
        following_right = (
            max(event["source_end_sample"] for event in following_events)
            if following_events
            else None
        )
        scoped_turns = [
            turn
            for turn in turns
            if left <= (turn["source_start_sample"] + turn["source_end_sample"]) // 2 < right
        ]
        following_turns = [
            turn
            for turn in turns
            if following_left is not None
            and following_right is not None
            and following_left
            <= (turn["source_start_sample"] + turn["source_end_sample"]) // 2
            < following_right
        ]
        scoped_tokens = [
            token for token in tokens if token["receipt_segment_id"] == segment_id
        ]
        expected = [
            part for item in scoped_reference for part in normalized_words(str(item["text"]))
        ]
        actual = concatenated_piece_runs(
            scoped_tokens, lambda token: token["source_kind"] != "prefix_context"
        )
        prefix_words = concatenated_piece_runs(
            scoped_tokens, lambda token: token["source_kind"] == "prefix_context"
        )
        seg_distance, seg_subs, seg_deletions, seg_insertions = edit_counts(expected, actual)
        segment_metrics.append(
            {
                "segment_id": segment_id,
                "source_span": [left, right],
                "following_segment_id": following_id,
                "boundary_type": finalize_by_segment[segment_id]["boundary_type"],
                "short_human_turn_count": sum(
                    turn["source_end_sample"] - turn["source_start_sample"] < RATE
                    for turn in scoped_turns
                ),
                "following_segment": {
                    "source_span": (
                        [following_left, following_right]
                        if following_left is not None and following_right is not None
                        else None
                    ),
                    "gap_ms": (
                        (following_left - right) * 1000 / RATE
                        if following_left is not None
                        else None
                    ),
                    "duration_ms": (
                        (following_right - following_left) * 1000 / RATE
                        if following_left is not None and following_right is not None
                        else None
                    ),
                    "human_turn_count": len(following_turns),
                    "short_human_turn_count": sum(
                        turn["source_end_sample"] - turn["source_start_sample"] < RATE
                        for turn in following_turns
                    ),
                },
                "prefix_context_token_pieces": sum(
                    token["source_kind"] == "prefix_context" for token in scoped_tokens
                ),
                "prefix_context_words": len(prefix_words),
                "reference_words": len(expected),
                "hypothesis_words": len(actual),
                "adjacent_duplicate_hypothesis_words": sum(
                    left_word == right_word
                    for left_word, right_word in zip(actual, actual[1:], strict=False)
                ),
                "distance": seg_distance,
                "substitutions": seg_subs,
                "deletions": seg_deletions,
                "insertions": seg_insertions,
                "wer": seg_distance / len(expected) if expected else None,
            }
        )
    receipt_reference_words = sum(metric["reference_words"] for metric in segment_metrics)
    receipt_hypothesis_words = sum(metric["hypothesis_words"] for metric in segment_metrics)
    receipt_distance = sum(metric["distance"] for metric in segment_metrics)
    receipt_substitutions = sum(metric["substitutions"] for metric in segment_metrics)
    receipt_deletions = sum(metric["deletions"] for metric in segment_metrics)
    receipt_insertions = sum(metric["insertions"] for metric in segment_metrics)
    unmapped_reasons: dict[str, int] = {}
    for token in tokens:
        if token["source_sample"] is None:
            reason = str(token["source_kind"])
            unmapped_reasons[reason] = unmapped_reasons.get(reason, 0) + 1
    boundary_strata: dict[str, dict[str, Any]] = {}
    for metric in segment_metrics:
        stratum = boundary_strata.setdefault(
            metric["boundary_type"],
            {
                "segments": 0,
                "reference_words": 0,
                "hypothesis_words": 0,
                "distance": 0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "short_human_turns": 0,
                "following_segments": 0,
                "following_short_human_turns": 0,
                "adjacent_duplicate_hypothesis_words": 0,
                "prefix_context_token_pieces": 0,
                "prefix_context_words": 0,
            },
        )
        stratum["segments"] += 1
        for key in (
            "reference_words",
            "hypothesis_words",
            "distance",
            "substitutions",
            "deletions",
            "insertions",
            "prefix_context_token_pieces",
            "prefix_context_words",
        ):
            stratum[key] += metric[key]
        stratum["short_human_turns"] += metric["short_human_turn_count"]
        stratum["adjacent_duplicate_hypothesis_words"] += metric[
            "adjacent_duplicate_hypothesis_words"
        ]
        if metric["following_segment"]["source_span"] is not None:
            stratum["following_segments"] += 1
            stratum["following_short_human_turns"] += metric["following_segment"][
                "short_human_turn_count"
            ]
    for stratum in boundary_strata.values():
        stratum["wer"] = (
            stratum["distance"] / stratum["reference_words"]
            if stratum["reference_words"]
            else None
        )

    provider_associations: dict[str, set[str]] = {}
    reference_associations: dict[str, set[str]] = {speaker: set() for speaker in speakers}
    unalignable_tokens = 0
    unknown_speaker_tokens = 0
    for token in mapped:
        if token["speaker"] == "unknown":
            unknown_speaker_tokens += 1
        matches = reference_at(reference, token["source_sample"])
        if not matches:
            unalignable_tokens += 1
            continue
        for match in matches:
            provider_associations.setdefault(token["speaker"], set()).add(match["speaker_id"])
            reference_associations.setdefault(match["speaker_id"], set()).add(token["speaker"])

    scoped_turns = [
        turn
        for turn in turns
        if any(
            left <= (turn["source_start_sample"] + turn["source_end_sample"]) // 2 < right
            for left, right in source_intervals
        )
    ]

    def dominant_provider(turn: dict[str, Any]) -> str | None:
        labels = [
            token["speaker"]
            for token in mapped
            if turn["source_start_sample"] <= token["source_sample"] < turn["source_end_sample"]
        ]
        if not labels:
            return None
        return max(set(labels), key=lambda label: (labels.count(label), label))

    returning_total = returning_assessable = returning_preserved = returning_correct = 0
    for index in range(len(scoped_turns) - 2):
        first, middle, returned = scoped_turns[index : index + 3]
        if (
            first["speaker_id"] != returned["speaker_id"]
            or first["speaker_id"] == middle["speaker_id"]
        ):
            continue
        returning_total += 1
        first_label = dominant_provider(first)
        returned_label = dominant_provider(returned)
        if first_label is None or returned_label is None:
            continue
        returning_assessable += 1
        returning_preserved += int(first_label == returned_label)
        returning_correct += int(
            mapping.get(first_label) == first["speaker_id"]
            and mapping.get(returned_label) == returned["speaker_id"]
        )

    short_distance = short_substitutions = short_deletions = short_insertions = 0
    short_reference_words = short_hypothesis_words = short_with_reference = 0
    short_turns = [
        turn
        for turn in scoped_turns
        if turn["source_end_sample"] - turn["source_start_sample"] < RATE
    ]
    for turn in short_turns:
        turn_reference = [
            item
            for item in reference
            if item["speaker_id"] == turn["speaker_id"]
            and turn["source_start_sample"]
            <= (item["source_start_sample"] + item["source_end_sample"]) // 2
            < turn["source_end_sample"]
        ]
        expected = [
            part for item in turn_reference for part in normalized_words(str(item["text"]))
        ]
        if expected:
            short_with_reference += 1
        turn_tokens = [
            token
            for token in mapped
            if turn["source_start_sample"] <= token["source_sample"] < turn["source_end_sample"]
            and mapping.get(token["speaker"]) == turn["speaker_id"]
        ]
        actual = normalized_words("".join(str(token["text"]) for token in turn_tokens))
        current = edit_counts(expected, actual)
        short_distance += current[0]
        short_substitutions += current[1]
        short_deletions += current[2]
        short_insertions += current[3]
        short_reference_words += len(expected)
        short_hypothesis_words += len(actual)

    adjacent_duplicate_words = sum(
        metric["adjacent_duplicate_hypothesis_words"] for metric in segment_metrics
    )
    duplicate_pair_denominator = sum(
        max(0, metric["hypothesis_words"] - 1) for metric in segment_metrics
    )
    return {
        "final_token_count": len(tokens),
        "source_mapped_token_count": len(source_mapped),
        "fixed_speaker_mapping": mapping,
        "reference_word_count": receipt_reference_words,
        "hypothesis_word_count": receipt_hypothesis_words,
        "word_error": {
            "ownership": "receipt_segment",
            "distance": receipt_distance,
            "substitutions": receipt_substitutions,
            "deletions": receipt_deletions,
            "insertions": receipt_insertions,
            "wer": (
                receipt_distance / receipt_reference_words
                if receipt_reference_words
                else None
            ),
        },
        "receipt_scoped_timestamp_word_error": {
            "distance": distance,
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "wer": distance / len(ref) if ref else None,
        },
        "unscoped_provider_timestamp_diagnostic": {
            "warning": (
                "diagnostic only: provider timestamp overruns may map a receipt token into the "
                "next segment; never use for segment ownership"
            ),
            "distance": unscoped_timestamp_edits[0],
            "substitutions": unscoped_timestamp_edits[1],
            "deletions": unscoped_timestamp_edits[2],
            "insertions": unscoped_timestamp_edits[3],
            "wer": (
                unscoped_timestamp_edits[0] / len(ref)
                if ref
                else None
            ),
        },
        "speaker": {
            "correct": speaker_correct,
            "total": speaker_total,
            "accuracy": speaker_correct / speaker_total if speaker_total else None,
            "overlap_correct": overlap_correct,
            "overlap_total": overlap_total,
            "sequential_correct": speaker_correct - overlap_correct,
            "sequential_total": speaker_total - overlap_total,
            "sequential_accuracy": (
                (speaker_correct - overlap_correct) / (speaker_total - overlap_total)
                if speaker_total - overlap_total
                else None
            ),
            "overlap_accuracy": (
                overlap_correct / overlap_total if overlap_total else None
            ),
        },
        "segment_text_metrics": segment_metrics,
        "boundary_strata": boundary_strata,
        "diarization_structure": {
            "provider_labels_with_multiple_reference_speakers": {
                label: sorted(values)
                for label, values in provider_associations.items()
                if len(values) > 1
            },
            "merge_candidate_label_count": sum(
                len(values) > 1 for values in provider_associations.values()
            ),
            "provider_label_denominator": len(provider_associations),
            "reference_speakers_with_multiple_provider_labels": {
                speaker: sorted(values)
                for speaker, values in reference_associations.items()
                if len(values) > 1
            },
            "split_candidate_reference_speaker_count": sum(
                len(values) > 1 for values in reference_associations.values()
            ),
            "reference_speaker_denominator": len(speakers),
            "reference_speakers_missing_from_fixed_mapping": sorted(
                set(speakers) - set(mapping.values())
            ),
            "unknown_speaker_tokens": unknown_speaker_tokens,
            "unalignable_source_mapped_tokens": unalignable_tokens,
            "returning_speaker_aba": {
                "human_triplets": returning_total,
                "assessable_triplets": returning_assessable,
                "same_provider_label_on_return": returning_preserved,
                "fixed_mapping_correct_on_both_appearances": returning_correct,
            },
            "short_human_turns_under_1s": {
                "turns": len(short_turns),
                "turns_with_reference_words": short_with_reference,
                "reference_words": short_reference_words,
                "hypothesis_words": short_hypothesis_words,
                "distance": short_distance,
                "substitutions": short_substitutions,
                "deletions": short_deletions,
                "insertions": short_insertions,
                "wer": (
                    short_distance / short_reference_words
                    if short_reference_words
                    else None
                ),
            },
            "adjacent_duplicate_hypothesis_words": {
                "count": adjacent_duplicate_words,
                "word_pair_denominator": duplicate_pair_denominator,
            },
        },
        "alignment_coverage": {
            "reference_words_in_planned_scope": len(reference_scope),
            "provider_tokens_source_mapped": len(source_mapped),
            "provider_tokens_scored_content": len(mapped),
            "provider_prefix_context_tokens": sum(
                token["source_kind"] == "prefix_context" for token in source_mapped
            ),
            "provider_prefix_context_words": sum(
                metric["prefix_context_words"] for metric in segment_metrics
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
        "provider_tokens_unmapped_by_reason": unmapped_reasons,
        "timing": facts,
        "concrete_source_failures": concrete,
    }


def evaluate_control_alignment(
    primary_plan: dict[str, Any],
    primary_tokens: list[dict[str, Any]],
    primary_facts: dict[str, Any],
    observer_tokens: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    speakers: list[str],
) -> dict[str, Any]:
    primary_segments = [
        event for event in primary_plan["events"] if event.get("event") == "finalize"
    ]
    primary_spans: dict[str, tuple[int, int]] = {}
    for finalize in primary_segments:
        segment_id = finalize["segment_id"]
        audio = [
            event
            for event in primary_plan["events"]
            if event.get("event") == "audio"
            and event.get("kind") == "real_content"
            and event.get("segment_id") == segment_id
        ]
        if audio:
            primary_spans[segment_id] = (
                min(event["source_start_sample"] for event in audio),
                max(event["source_end_sample"] for event in audio),
            )
    intervals = list(primary_spans.values())
    observer_scope = sorted(
        [
            token
            for token in observer_tokens
            if token["source_sample"] is not None
            and token["source_kind"] == "continuous_observation"
            and any(left <= token["source_sample"] < right for left, right in intervals)
        ],
        key=lambda token: token["source_sample"],
    )
    reference_scope = [
        item
        for item in reference
        if any(
            left <= (item["source_start_sample"] + item["source_end_sample"]) // 2 < right
            for left, right in intervals
        )
    ]
    expected = [
        part for item in reference_scope for part in normalized_words(str(item["text"]))
    ]
    observer_hypothesis = normalized_words(
        "".join(str(token["text"]) for token in observer_scope)
    )
    observer_edits = edit_counts(expected, observer_hypothesis)
    primary_edits = [0, 0, 0, 0]
    primary_reference_word_count = 0
    primary_hypothesis_word_count = 0
    for segment_id, (left, right) in primary_spans.items():
        segment_expected = [
            part
            for item in reference
            if left <= (item["source_start_sample"] + item["source_end_sample"]) // 2 < right
            for part in normalized_words(str(item["text"]))
        ]
        segment_tokens = [
            token
            for token in primary_tokens
            if token["receipt_segment_id"] == segment_id
        ]
        segment_hypothesis = concatenated_piece_runs(
            segment_tokens, lambda token: token["source_kind"] != "prefix_context"
        )
        segment_edits = edit_counts(segment_expected, segment_hypothesis)
        primary_edits = [
            total + current
            for total, current in zip(primary_edits, segment_edits, strict=True)
        ]
        primary_reference_word_count += len(segment_expected)
        primary_hypothesis_word_count += len(segment_hypothesis)
    observer_mapping = fixed_mapping(observer_scope, reference, speakers)
    primary_speaker_scope = [
        token
        for token in primary_tokens
        if token["receipt_segment_id"] in primary_spans
        and token["source_kind"] != "prefix_context"
    ]
    primary_mapping = fixed_mapping(
        [
            token
            for token in primary_speaker_scope
            if token["source_sample"] is not None and token["source_kind"] == "real_content"
        ],
        reference,
        speakers,
    )

    def speaker_status(token: dict[str, Any], mapping: dict[str, str]) -> str:
        if token["speaker"] == "unknown":
            return "unknown"
        if token["source_sample"] is None:
            return "unalignable"
        matches = reference_at(reference, token["source_sample"])
        if not matches:
            return "unalignable"
        expected_speakers = {item["speaker_id"] for item in matches}
        if len(expected_speakers) > 1:
            return "mixed"
        return "correct" if mapping.get(token["speaker"]) in expected_speakers else "incorrect"

    totals = {
        "correct": 0,
        "incorrect": 0,
        "unknown": 0,
        "mixed": 0,
        "unalignable": 0,
    }
    primary_totals = {key: 0 for key in totals}
    for token in observer_scope:
        totals[speaker_status(token, observer_mapping)] += 1
    for token in primary_speaker_scope:
        primary_totals[speaker_status(token, primary_mapping)] += 1

    segments: list[dict[str, Any]] = []
    last_label_delays: list[float] = []
    for finalize in primary_segments:
        segment_id = finalize["segment_id"]
        if segment_id not in primary_spans:
            continue
        left, right = primary_spans[segment_id]
        ready = primary_facts.get("segments", {}).get(segment_id, {}).get(
            "primary_ready_offset_ms"
        )
        scoped = [
            token for token in observer_scope if left <= token["source_sample"] < right
        ]
        available = (
            [token for token in scoped if token["receipt_offset_ms"] <= ready]
            if ready is not None
            else []
        )
        scoped_reference = [
            item
            for item in reference
            if left <= (item["source_start_sample"] + item["source_end_sample"]) // 2 < right
        ]
        scoped_expected = [
            part for item in scoped_reference for part in normalized_words(str(item["text"]))
        ]
        partial_hypothesis = normalized_words(
            "".join(str(token["text"]) for token in available)
        )
        partial_edits = edit_counts(scoped_expected, partial_hypothesis)
        available_status = {key: 0 for key in totals}
        later_status = {key: 0 for key in totals}
        for token in available:
            available_status[speaker_status(token, observer_mapping)] += 1
        for token in scoped:
            later_status[speaker_status(token, observer_mapping)] += 1
        label_receipts = [
            token["receipt_offset_ms"] for token in scoped if token["speaker"] != "unknown"
        ]
        delay = (
            max(label_receipts) - ready
            if ready is not None and label_receipts
            else None
        )
        if delay is not None:
            last_label_delays.append(delay)
        segments.append(
            {
                "segment_id": segment_id,
                "authoritative_primary_source_span": [left, right],
                "primary_ready_offset_ms": ready,
                "observer_token_pieces_available_at_primary_ready": len(available),
                "observer_token_pieces_available_later": len(scoped),
                "speaker_status_at_primary_ready": available_status,
                "speaker_status_later": later_status,
                "available_partial_text": {
                    "hypothesis_words": len(partial_hypothesis),
                    "reference_words": len(scoped_expected),
                    "distance": partial_edits[0],
                    "wer": (
                        partial_edits[0] / len(scoped_expected)
                        if scoped_expected
                        else None
                    ),
                },
                "last_observer_label_minus_primary_ready_ms": delay,
            }
        )
    ordered_delays = sorted(last_label_delays)
    ready_status_totals = {key: 0 for key in totals}
    for segment in segments:
        for key, value in segment["speaker_status_at_primary_ready"].items():
            ready_status_totals[key] += value
    ready_token_pieces = sum(
        segment["observer_token_pieces_available_at_primary_ready"] for segment in segments
    )
    return {
        "observer_role": (
            "annotation-only; primary source spans and primary text remain authoritative, and "
            "observer text is never substituted"
        ),
        "fixed_observer_speaker_mapping": observer_mapping,
        "speaker_status_later": totals,
        "speaker_status_at_primary_ready": ready_status_totals,
        "observer_token_pieces_at_primary_ready": ready_token_pieces,
        "observer_token_pieces_later": len(observer_scope),
        "speaker_accuracy_excluding_unknown_mixed_unalignable": {
            "correct": totals["correct"],
            "total": totals["correct"] + totals["incorrect"],
            "accuracy": (
                totals["correct"] / (totals["correct"] + totals["incorrect"])
                if totals["correct"] + totals["incorrect"]
                else None
            ),
        },
        "fixed_primary_speaker_mapping": primary_mapping,
        "primary_speaker_status": primary_totals,
        "primary_speaker_accuracy_excluding_unknown_mixed_unalignable": {
            "correct": primary_totals["correct"],
            "total": primary_totals["correct"] + primary_totals["incorrect"],
            "accuracy": (
                primary_totals["correct"]
                / (primary_totals["correct"] + primary_totals["incorrect"])
                if primary_totals["correct"] + primary_totals["incorrect"]
                else None
            ),
        },
        "observer_scope_matched_word_error": {
            "reference_words": len(expected),
            "hypothesis_words": len(observer_hypothesis),
            "distance": observer_edits[0],
            "substitutions": observer_edits[1],
            "deletions": observer_edits[2],
            "insertions": observer_edits[3],
            "wer": observer_edits[0] / len(expected) if expected else None,
        },
        "primary_same_span_word_error": {
            "ownership": "receipt_segment",
            "reference_words": primary_reference_word_count,
            "hypothesis_words": primary_hypothesis_word_count,
            "distance": primary_edits[0],
            "substitutions": primary_edits[1],
            "deletions": primary_edits[2],
            "insertions": primary_edits[3],
            "wer": (
                primary_edits[0] / primary_reference_word_count
                if primary_reference_word_count
                else None
            ),
        },
        "last_label_minus_primary_ready_ms": {
            "count": len(ordered_delays),
            "min": ordered_delays[0] if ordered_delays else None,
            "median": (
                (
                    ordered_delays[(len(ordered_delays) - 1) // 2]
                    + ordered_delays[len(ordered_delays) // 2]
                )
                / 2
                if ordered_delays
                else None
            ),
            "max": ordered_delays[-1] if ordered_delays else None,
        },
        "segments": segments,
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
    loaded: dict[tuple[str, str, str], tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[str]]] = {}
    missing_traces: list[str] = []
    for plan in plan_payload["plans"]:
        recording = by_recording[plan["recording_id"]]
        reference, speakers, turns = reference_words(args.manifest.parent / recording["human_reference"]["path"])
        trace_path = args.run_dir / f"{plan['recording_id']}--{plan['arm']}--{plan['stream_role']}.jsonl"
        if not trace_path.is_file():
            missing_traces.append(trace_path.name)
            continue
        tokens, facts = load_trace(trace_path, plan)
        loaded[(plan["recording_id"], plan["arm"], plan["stream_role"])] = (
            plan,
            tokens,
            facts,
            reference,
            speakers,
        )
        results.append(
            {
                "recording_id": plan["recording_id"],
                "arm": plan["arm"],
                "stream_role": plan["stream_role"],
                "metrics": evaluate_stream(tokens, facts, reference, speakers, turns, plan),
            }
        )
    control_alignments: list[dict[str, Any]] = []
    control_recordings = sorted(
        recording_id
        for recording_id, arm, role in loaded
        if arm == "C" and role == "primary"
    )
    for recording_id in control_recordings:
        primary = loaded.get((recording_id, "C", "primary"))
        observer = loaded.get((recording_id, "C", "observer"))
        if primary is None or observer is None:
            continue
        primary_plan, primary_tokens, primary_facts, reference, speakers = primary
        _, observer_tokens, _, _, _ = observer
        control_alignments.append(
            {
                "recording_id": recording_id,
                "arm": "C",
                "metrics": evaluate_control_alignment(
                    primary_plan,
                    primary_tokens,
                    primary_facts,
                    observer_tokens,
                    reference,
                    speakers,
                ),
            }
        )
    output = {
        "status": "evaluated",
        "execution_identity": plan_payload["execution_identity"],
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coverage": plan_payload["coverage"],
        "streams": results,
        "missing_traces": missing_traces,
        "control_observer_alignment": control_alignments,
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
