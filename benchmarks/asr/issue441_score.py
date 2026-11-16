from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

try:
    from benchmarks.asr.issue373_score import load, timing
except ModuleNotFoundError:
    from issue373_score import load, timing


ARMS = ("N", "S0", "S1", "E0", "T1", "T2")
MAPPING_METHOD = (
    "Maximum correct fixed anchors in eventual receive-ordered state, one-to-one per "
    "recording/session; deterministic lexical tie break. Reused unchanged online. "
    "Only eventual canonical IDs enter the mapping. Online raw IDs use only merges "
    "received by that anchor cutoff; earlier fragmented/unmatched IDs remain wrong."
)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def lexical(text):
    return any(char.isalnum() for char in str(text or ""))


def normalized(text):
    return "".join(
        char for char in unicodedata.normalize("NFKC", str(text)).casefold() if char.isalnum()
    )


def source_lexical_unit(unit):
    raw = unit.get("raw") or {}
    return lexical(unit.get("text")) and raw.get("type") not in {"spacing", "audio_event"}


def compact_source_text(text):
    return "".join(char for char in unicodedata.normalize("NFKC", str(text)) if not char.isspace())


def current_composition_projection(units):
    groups = []
    leading = ""
    for unit in units:
        raw = unit.get("raw") or {}
        if raw.get("type") in {"spacing", "audio_event"}:
            continue
        text = compact_source_text(unit.get("text", ""))
        if not lexical(text):
            if groups:
                groups[-1]["normalized_source_text"] += text
            else:
                leading += text
            continue
        text, leading = leading + text, ""
        scope = unit.get("session")
        language = unit.get("language")
        identity = speaker(unit.get("speaker"))
        previous = groups[-1] if groups else None
        if (
            previous is not None
            and previous["scope"] == scope
            and previous["language"] == language
            and (
                previous["representative"] is None
                or identity is None
                or previous["representative"] == identity
            )
        ):
            previous["normalized_source_text"] += text
            if previous["representative"] is None:
                previous["representative"] = identity
        else:
            groups.append(
                {
                    "normalized_source_text": text,
                    "scope": scope,
                    "language": language,
                    "representative": identity,
                }
            )
    return groups


def lexical_reference_matches(pieces, reference):
    observed = "".join(pieces)
    boundaries = {0, *itertools.accumulate(map(len, pieces))}
    result = []
    position = observed.find(reference) if reference else -1
    while position >= 0:
        end = position + len(reference)
        if position in boundaries and end in boundaries:
            result.append(end)
        position = observed.find(reference, position + 1)
    return result


def ending_completion(words, reference_text, ending_s):
    ordered = sorted(words, key=lambda row: (row[1]["start_s"], row[1]["end_s"]))
    pieces = [normalized(word["text"]) for _, word in ordered]
    reference = normalized(reference_text or "")
    ends = list(itertools.accumulate(map(len, pieces)))
    matches = lexical_reference_matches(pieces, reference)
    if matches:
        index = next(index for index, endpoint in enumerate(ends) if endpoint == matches[0])
        _, word = ordered[index]
        start = matches[0] - len(reference)
        participating = {
            id(origin): origin
            for position, (origin, _) in enumerate(ordered[: index + 1])
            if ends[position] > start
        }
        revised = [
            origin
            for origin in participating.values()
            if not origin["committed_source_text_unchanged"]
        ]
        if revised:
            return {
                "status": "later_text_revision_not_backdated",
                "delay_s": None,
                "revised_segments": [
                    {
                        key: origin.get(key)
                        for key in (
                            "segment_index",
                            "text",
                            "eventual_source_text",
                            "arrival_s",
                            "eventual_source_arrival_s",
                            "evidence",
                        )
                    }
                    for origin in revised
                ],
                "support": "At least one participating segment's eventual source text differs from its original committed receipt. Later restored text is not credited to that earlier commit.",
            }
        commit = max(participating.values(), key=lambda origin: origin["arrival_s"])
        return {
            "status": "supported",
            "delay_s": commit["arrival_s"] - ending_s,
            "commit_evidence": commit["evidence"],
            "ending_word_end_s": word["end_s"],
            "support": "Full frozen reference ending has timed lexical endpoint support; every participating segment's source text matches its original committed receipt. Credits the latest participating commit receipt, not delayed alignment or restored text.",
        }
    return {
        "status": (
            "reference_ending_content_not_represented"
            if reference
            else "reference_text_unavailable"
        ),
        "delay_s": None,
        "observed_reference_span_text": "".join(word["text"] for _, word in ordered),
        "support": "No completion latency assigned without full frozen reference-text endpoint support; substitutions, missing content, and unlocalizable timing remain unavailable.",
    }


def speaker(value):
    return None if value is None or value == "" else str(value)


def canonical(value, merges):
    seen = set()
    while value in merges and value not in seen:
        seen.add(value)
        value = merges[value]
    return value


def optimal_mapping(weights, references):
    refs = sorted(references)
    states = {0: (0, {})}
    for native in sorted({native for native, _ in weights}):
        updated = dict(states)
        for mask, (score, mapping) in states.items():
            for index, ref in enumerate(refs):
                weight = weights[native, ref]
                if mask & (1 << index) or not weight:
                    continue
                key = mask | (1 << index)
                candidate = score + weight
                if key not in updated or candidate > updated[key][0]:
                    updated[key] = candidate, {**mapping, native: ref}
        states = updated
    return max(states.values(), key=lambda item: item[0])[1]


def merge_pairs(raw):
    if isinstance(raw.get("merges"), list):
        return [pair for item in raw["merges"] for pair in merge_pairs(item)]
    if isinstance(raw.get("speaker_id_mapping"), dict):
        return [(str(a), str(b)) for a, b in raw["speaker_id_mapping"].items()]
    target = raw.get("to_speaker_id", raw.get("merged_into", raw.get("target_speaker_id")))
    sources = raw.get("speaker_ids", raw.get("source_speaker_ids"))
    if sources is not None and target is not None:
        return [(str(item), str(target)) for item in sources if str(item) != str(target)]
    source = raw.get("from_speaker_id", raw.get("speaker_id", raw.get("source_speaker_id")))
    return [(str(source), str(target))] if source is not None and target is not None else []


def word_units(words, row, session="0"):
    units = []
    for index, word in enumerate(words or []):
        if word.get("type") in {"spacing", "audio_event"} or not str(word.get("text", "")).strip():
            continue
        start = word.get("start", word.get("start_s"))
        end = word.get("end", word.get("end_s"))
        if "start_ms" in word:
            start = word["start_ms"] / 1000 if number(word["start_ms"]) else None
            end = word["end_ms"] / 1000 if number(word.get("end_ms")) else None
        units.append(
            {
                "start_s": start,
                "end_s": end,
                "text": word.get("text", ""),
                "speaker": speaker(word.get("speaker_id", word.get("speaker"))),
                "settled": bool(row.get("settled", False)),
                "arrival_s": row["arrival_s"],
                "segment_index": row.get("segment_index"),
                "session": str(session),
                "evidence": row.get("evidence"),
                "word_index": index,
            }
        )
    return units


def to_source(unit, run):
    if run["arm"] != "S1" or unit.get("time_domain") == "source":
        return [unit]
    a, b = unit["start_s"], unit["end_s"]
    if not number(a) or not number(b):
        return [unit]
    pieces = []
    covered = 0.0
    for sent in run.get("sends", []):
        if sent.get("status") not in {"completed", "sent", "ok", "success"}:
            continue
        if str(sent.get("session_id", sent.get("session", "0"))) != unit["session"]:
            continue
        x, y = sent.get("transmitted_start_s"), sent.get("transmitted_end_s")
        if not number(x) or not number(y) or y <= x:
            continue
        ranges = sent.get("source_ranges")
        if ranges is None:
            ranges = [{"start_s": sent.get("source_start_s"), "end_s": sent.get("source_end_s")}]
        cursor = x
        for span in ranges:
            lo, hi = span.get("start_s"), span.get("end_s")
            if not number(lo) or not number(hi):
                continue
            count = hi - lo
            left, right = max(cursor, a), min(cursor + count, b)
            if left < right:
                pieces.append({**unit, "start_s": lo + left - cursor, "end_s": lo + right - cursor})
                covered += right - left
            cursor += count
    contiguous = all(
        abs(left["end_s"] - right["start_s"]) < 1e-5 for left, right in zip(pieces, pieces[1:])
    )
    if not pieces or not contiguous or abs(covered - (b - a)) >= 1e-4:
        return [{**unit, "start_s": None, "end_s": None}]
    return [{**unit, "start_s": pieces[0]["start_s"], "end_s": pieces[-1]["end_s"]}]


class Journal:
    def __init__(self, run, evidence=""):
        self.run = run
        self.evidence = evidence
        self.events = sorted(
            run.get("events", []), key=lambda row: (row["arrival_s"], row.get("seq", 0))
        )
        self.cache = {}
        self.issues = []
        self.raw_native = any(
            isinstance(row.get("raw"), dict)
            and (
                "tokens" in row["raw"]
                or row["raw"].get("message_type")
                in {
                    "partial_transcript",
                    "committed_transcript",
                    "committed_transcript_with_timestamps",
                    "diarized_transcript",
                    "speakers_merged",
                }
            )
            for row in self.events
        )
        self.activity = (
            [
                {
                    **row,
                    "session": str(row.get("session_id", row.get("session", "0"))),
                    "speaker": speaker(row.get("speaker")),
                    "time_domain": "source",
                    "evidence": f"{evidence}#observations/{index}",
                }
                for index, row in enumerate(run.get("observations", []))
            ]
            if run["arm"] == "N"
            else None
        )

    def exported_snapshot(self, cutoff):
        active, latest = {}, {}
        for snapshot in sorted(
            self.run["snapshots"], key=lambda row: (row["arrival_s"], row["revision"])
        ):
            if snapshot["arrival_s"] > cutoff:
                break
            key = str(snapshot.get("session_id", "0")), snapshot["snapshot_key"]
            operation = snapshot["operation"]
            latest[key] = snapshot
            if operation == "retract":
                active.pop(key, None)
                latest.pop(key, None)
            elif operation == "append":
                active.setdefault(key, []).extend(snapshot["observation_indices"])
            elif operation == "replace":
                active[key] = list(snapshot["observation_indices"])
            else:
                raise ValueError(f"Unsupported snapshot operation: {operation}")
        rows = []
        for (session, key), indices in active.items():
            if not indices:
                snapshot = latest[session, key]
                event = next(
                    (row for row in self.events if row["seq"] == snapshot["event_seq"]), None
                )
                raw = event.get("raw", {}) if event else {}
                if lexical(raw.get("text")):
                    rows.extend(
                        {**unit, "snapshot_key": key, "time_domain": "source"}
                        for unit in word_units(
                            [{"text": raw["text"]}],
                            {
                                **snapshot,
                                "evidence": f"{self.evidence}#events/{snapshot['event_seq']}",
                            },
                            session,
                        )
                    )
            for index in indices:
                row = self.run["observations"][index]
                if row["arrival_s"] > cutoff:
                    raise ValueError("Snapshot references a future observation")
                rows.append(
                    {
                        **row,
                        "session": session,
                        "snapshot_key": key,
                        "speaker": speaker(row.get("speaker")),
                        "time_domain": "source",
                        "evidence": f"{self.evidence}#observations/{index}",
                    }
                )
        priorities = {}
        for row in rows:
            index = row.get("segment_index")
            if index is not None:
                priority = (
                    2
                    if "diarized" in row["snapshot_key"]
                    else 1 if "with_timestamps" in row["snapshot_key"] else 0
                )
                key = row["session"], index
                priorities[key] = max(priorities.get(key, -1), priority)
        filtered = []
        for row in rows:
            index = row.get("segment_index")
            priority = (
                2
                if "diarized" in row["snapshot_key"]
                else 1 if "with_timestamps" in row["snapshot_key"] else 0
            )
            if index is None or priority == priorities[row["session"], index]:
                filtered.append(row)
        merges = defaultdict(dict)
        for event in self.events:
            if event["arrival_s"] > cutoff:
                break
            raw = event.get("raw", {})
            if raw.get("message_type") == "speakers_merged":
                session = str(event.get("session_id", "0"))
                for old, new in merge_pairs(raw):
                    merges[session][old] = canonical(new, merges[session])
        return [
            {**row, "speaker": canonical(row["speaker"], merges[row["session"]])}
            for row in filtered
        ]

    def snapshot(self, cutoff=math.inf):
        if cutoff in self.cache:
            return self.cache[cutoff]
        if self.activity is not None:
            return [row for row in self.activity if row["arrival_s"] <= cutoff]
        if self.run.get("snapshots"):
            result = self.exported_snapshot(cutoff)
            self.cache[cutoff] = result
            return result
        segments, drafts, finals, merges = {}, {}, defaultdict(list), defaultdict(dict)
        priorities = {}
        if self.raw_native:
            for row in self.events:
                if row["arrival_s"] > cutoff:
                    break
                raw = row.get("raw", {})
                if not isinstance(raw, dict):
                    continue
                session = str(row.get("session_id", row.get("session", "0")))
                base = {**row, "evidence": f"{self.evidence}#events/{row['seq']}"}
                if "tokens" in raw:
                    tokens = [
                        item
                        for item in raw["tokens"]
                        if lexical(item.get("text")) and not item.get("text", "").startswith("<")
                    ]
                    settled = [item for item in tokens if item.get("is_final")]
                    provisional = [item for item in tokens if not item.get("is_final")]
                    finals[session].extend(word_units(settled, {**base, "settled": True}, session))
                    drafts[session] = word_units(provisional, base, session)
                    continue
                kind = raw.get("message_type", row.get("kind"))
                if kind == "speakers_merged":
                    for old, new in merge_pairs(raw):
                        merges[session][old] = canonical(new, merges[session])
                    continue
                index = raw.get("segment_index")
                if kind == "partial_transcript":
                    units = word_units(raw.get("words"), {**base, "segment_index": index}, session)
                    if not units and lexical(raw.get("text")):
                        units = word_units([{"text": raw["text"]}], base, session)
                    drafts[session] = units
                elif kind in {
                    "committed_transcript",
                    "committed_transcript_with_timestamps",
                    "diarized_transcript",
                }:
                    if index is None:
                        key = f"unindexed:{row['seq']}"
                    else:
                        key = index
                    priority = {
                        "committed_transcript": 0,
                        "committed_transcript_with_timestamps": 1,
                        "diarized_transcript": 2,
                    }[kind]
                    if priority < priorities.get((session, key), -1):
                        if kind == "committed_transcript":
                            drafts[session] = []
                        continue
                    priorities[session, key] = priority
                    old = segments.get((session, key), [])
                    units = word_units(
                        raw.get("words"),
                        {**base, "segment_index": index, "settled": kind != "committed_transcript"},
                        session,
                    )
                    if not units and lexical(raw.get("text")):
                        units = word_units(
                            [{"text": raw["text"]}], {**base, "segment_index": index}, session
                        )
                    if kind == "committed_transcript" and old:
                        units = old
                    segments[session, key] = units
                    if kind == "committed_transcript":
                        drafts[session] = []
        else:
            grouped = defaultdict(list)
            for index, observation in enumerate(self.run.get("observations", [])):
                if observation["arrival_s"] > cutoff:
                    continue
                key = (
                    str(observation.get("session_id", observation.get("session", "0"))),
                    observation.get("segment_index", observation.get("id", index)),
                )
                grouped[key].append(
                    {**observation, "evidence": f"{self.evidence}#observations/{index}"}
                )
            for key, rows in grouped.items():
                latest = max(row["arrival_s"] for row in rows)
                segments[key] = [
                    {
                        **row,
                        "session": key[0],
                        "speaker": speaker(row.get("speaker")),
                        "time_domain": "source",
                    }
                    for row in rows
                    if row["arrival_s"] == latest
                ]
        units = [
            unit
            for rows in itertools.chain(segments.values(), finals.values(), drafts.values())
            for unit in rows
        ]
        mapped = [piece for unit in units for piece in to_source(unit, self.run)]
        result = [
            {**unit, "speaker": canonical(unit["speaker"], merges[unit["session"]])}
            for unit in mapped
        ]
        self.cache[cutoff] = result
        return result

    def commits(self):
        result, indexed, previous = [], {}, {}
        for row in self.events:
            raw = row.get("raw", {})
            if not isinstance(raw, dict) or raw.get("message_type") != "committed_transcript":
                continue
            index, session = raw.get("segment_index"), str(
                row.get("session_id", row.get("session", "0"))
            )
            if index is not None and (session, index) in indexed:
                continue
            commit = {
                "arrival_s": row["arrival_s"],
                "segment_index": index,
                "session": session,
                "text": raw.get("text", ""),
                "evidence": f"{self.evidence}#events/{row.get('seq')}",
                "cause": raw.get("commit_reason", row.get("commit_reason", "unavailable")),
            }
            commit["event_seq"] = row.get("seq")
            commit["previous_native_commit_seq"] = previous.get(session)
            previous[session] = row.get("seq")
            if index is not None:
                indexed[session, index] = commit
            result.append(commit)
        for commit in result:
            units = (
                [
                    unit
                    for unit in self.snapshot()
                    if unit.get("segment_index") == commit["segment_index"]
                    and unit["session"] == commit["session"]
                ]
                if commit["segment_index"] is not None
                else []
            )
            timed = [
                unit
                for unit in units
                if source_lexical_unit(unit)
                and number(unit.get("start_s"))
                and number(unit.get("end_s"))
            ]
            commit["start_s"] = min((unit["start_s"] for unit in timed), default=None)
            commit["end_s"] = max((unit["end_s"] for unit in timed), default=None)
            commit["words"] = timed
            commit["eventual_source_text"] = "".join(
                group["normalized_source_text"] for group in current_composition_projection(units)
            )
            commit["committed_source_text_unchanged"] = (
                compact_source_text(commit["text"]) == commit["eventual_source_text"]
            )
            commit["eventual_source_arrival_s"] = max(
                (unit["arrival_s"] for unit in units), default=None
            )
            controls = [
                row
                for row in self.events
                if row.get("kind") == "native_commit_control_state"
                and row.get("raw", {}).get("segment_index") == commit["segment_index"]
                and str(row.get("session_id", "0")) == commit["session"]
            ]
            if controls:
                commit["cause"] = controls[0]["raw"].get("cause", commit["cause"])
                commit["control_state"] = controls[0]["raw"]
            commit["possible_manual_commands"] = self.manual_command_contexts(commit, result)
            matching = [
                item
                for item in self.run.get("commits", [])
                if item.get("segment_index") == commit["segment_index"]
                and commit["segment_index"] is not None
            ]
            if matching:
                commit["cause"] = matching[0].get(
                    "cause", matching[0].get("reason", commit["cause"])
                )
            siblings = [row for row in result if row["session"] == commit["session"]]
            commit["first_in_session"] = commit is siblings[0]
            commit["last_in_session"] = commit is siblings[-1]
            commit["empty_text"] = not lexical(commit["text"])
        return result

    def manual_command_contexts(self, commit, commits):
        contexts = []
        for index, sent in enumerate(self.run.get("sends", [])):
            command = sent.get("control") or {}
            reason = command.get("reason", sent.get("kind", "").removesuffix("_commit"))
            if (
                reason not in {"cap", "eof"}
                or str(sent.get("session_id", "0")) != commit["session"]
            ):
                continue
            start, end = sent.get("start_s"), sent.get("completed_s")
            if not number(start) or commit["arrival_s"] < start:
                continue
            during = not number(end) or commit["arrival_s"] <= end
            after = [
                row
                for row in commits
                if row["session"] == commit["session"] and number(end) and row["arrival_s"] >= end
            ]
            first_after = bool(after) and min(after, key=lambda row: row["arrival_s"]) is commit
            predecessor = (
                "last_native_commit_seq" in command
                and command["last_native_commit_seq"] == commit["previous_native_commit_seq"]
            )
            if during or first_after or predecessor:
                contexts.append(
                    {
                        **command,
                        "reason": reason,
                        "send_start_s": start,
                        "send_completed_s": end,
                        "association": (
                            "in_flight_send"
                            if during
                            else (
                                "first_receipt_after_send"
                                if first_after
                                else "recorded_command_predecessor"
                            )
                        ),
                        "evidence": f"{self.evidence}#sends/{index}",
                    }
                )
        command = (commit.get("control_state") or {}).get("last_command") or {}
        if (
            command
            and command.get("last_native_commit_seq") == commit["previous_native_commit_seq"]
        ):
            contexts.append({**command, "association": "receipt_control_state_predecessor"})
        return contexts


def at_anchor(units, anchor, recording, activity=False):
    t = anchor["time_s"]
    timed = [
        unit
        for unit in units
        if number(unit.get("start_s"))
        and number(unit.get("end_s"))
        and unit["start_s"] <= t < unit["end_s"]
    ]
    usable = [unit for unit in timed if activity or lexical(unit.get("text"))]
    assigned = [unit for unit in usable if unit.get("speaker") is not None]
    if assigned:
        newest = max(unit["arrival_s"] for unit in assigned)
        selected = [unit for unit in assigned if unit["arrival_s"] == newest]
        identities = {(unit["session"], unit["speaker"]) for unit in selected}
        if len(identities) == 1:
            return {
                "state": "assigned",
                "session": selected[0]["session"],
                "speaker": selected[0]["speaker"],
                "settled": all(unit.get("settled", False) for unit in selected),
                "evidence": selected[0].get("evidence"),
            }
        return {"state": "missing_speaker", "detail": "conflicting_simultaneous_ids"}
    if usable:
        return {"state": "missing_speaker"}
    if activity:
        return {"state": "missing_speaker", "detail": "no_usable_activity_identity_at_anchor"}
    text_units = [unit for unit in units if source_lexical_unit(unit)]
    untimed = [
        unit
        for unit in text_units
        if not number(unit.get("start_s"))
        or not number(unit.get("end_s"))
        or unit["end_s"] <= unit["start_s"]
    ]
    local_untimed = [
        unit
        for unit in untimed
        if any(
            number(source.get("start_s"))
            and number(source.get("end_s"))
            and source["start_s"] <= t < source["end_s"]
            for source in unit.get("source_ranges", [])
        )
    ]
    if local_untimed:
        return {
            "state": "missing_timing",
            "detail": "explicit_source_range_covers_anchor_but_word_alignment_unusable",
            "evidence": local_untimed[0].get("evidence"),
        }
    if not text_units:
        return {
            "state": "missing_text",
            "detail": "no_lexical_text_received_by_cutoff",
            "reference_span": anchor["span_id"],
        }
    return {
        "state": "no_timed_text_coverage",
        "detail": "text_exists_but_anchor_local_cause_unsupported",
        "session_has_untimed_text": bool(untimed),
    }


def counts(rows):
    result = dict(Counter(row["outcome"] for row in rows))
    for reason in ("missing_text", "missing_timing", "missing_speaker", "no_timed_text_coverage"):
        result.setdefault(reason, 0)
    n = len(rows)
    assigned = result.get("correct", 0) + result.get("wrong", 0)
    result.update(
        {
            "eligible": n,
            "assigned": assigned,
            "correct": result.get("correct", 0),
            "wrong": result.get("wrong", 0),
            "unassigned": n - assigned,
            "assigned_coverage": assigned / n if n else None,
            "conditional_accuracy": result.get("correct", 0) / assigned if assigned else None,
            "overall_correct": result.get("correct", 0) / n if n else None,
            "provisional_assigned": sum(
                row.get("outcome") in {"correct", "wrong"} and not row.get("settled")
                for row in rows
            ),
            "settled_assigned": sum(
                row.get("outcome") in {"correct", "wrong"} and row.get("settled", False)
                for row in rows
            ),
        }
    )
    return result


def connected(run):
    if "connected" in run:
        return bool(run["connected"])
    if run.get("status") in {"unrun", "unavailable", "connection_failed", "setup_failed"}:
        return False
    return bool(
        run.get("observations")
        or run.get("sends")
        or any(
            row.get("kind") in {"connected", "session_started"}
            or row.get("raw", {}).get("message_type") == "session_started"
            for row in run.get("events", [])
            if isinstance(row.get("raw", {}), dict)
        )
    )


def score_recording(recording, run, evidence=""):
    if run is None or not connected(run):
        return {
            "recording_id": recording["id"],
            "language": recording["language"],
            "status": "unrun",
            "planned_anchors": len(recording["anchors"]),
            "reason": "no_authorized_connected_session",
            "planned_turn_opportunities": len(recording.get("turn_opportunities", [])),
            "run_status": run.get("status") if run else "missing_log",
            "errors": run.get("errors", []) if run else [],
        }
    if run.get("metadata", {}).get("input_sha256") != recording["sha256"]:
        raise ValueError(f"Input hash mismatch: {run.get('arm')}/{recording['id']}")
    journal = Journal(run, evidence)
    activity = run["arm"] == "N"
    anchors = recording["anchors"]
    eventual = [at_anchor(journal.snapshot(), anchor, recording, activity) for anchor in anchors]
    weights = defaultdict(Counter)
    for anchor, value in zip(anchors, eventual, strict=True):
        if value["state"] == "assigned":
            weights[value["session"]][value["speaker"], str(anchor["speaker"])] += 1
    refs = {str(anchor["speaker"]) for anchor in anchors}
    mapping = {session: optimal_mapping(weight, refs) for session, weight in weights.items()}
    views = {}
    for name in ("online", "eventual"):
        rows = []
        for index, anchor in enumerate(anchors):
            cutoff = anchor["span_end_s"] + 1.0
            value = (
                eventual[index]
                if name == "eventual"
                else at_anchor(journal.snapshot(cutoff), anchor, recording, activity)
            )
            outcome = value["state"]
            if outcome == "assigned":
                outcome = (
                    "correct"
                    if mapping.get(value["session"], {}).get(value["speaker"])
                    == str(anchor["speaker"])
                    else "wrong"
                )
            rows.append(
                {
                    "anchor_id": anchor["id"],
                    "time_s": anchor["time_s"],
                    "reference": anchor["speaker"],
                    "cutoff_s": cutoff if name == "online" else None,
                    **value,
                    "outcome": outcome,
                    "unmatched_id": value["state"] == "assigned"
                    and value["speaker"] not in mapping.get(value["session"], {}),
                }
            )
        views[name] = {"counts": counts(rows), "anchors": rows}
    overlaps = []
    for overlap in recording.get("overlaps", []):
        for name, cutoff in (("online", overlap["end_s"] + 1), ("eventual", math.inf)):
            units = [
                unit
                for unit in journal.snapshot(cutoff)
                if number(unit.get("start_s"))
                and number(unit.get("end_s"))
                and max(unit["start_s"], overlap["start_s"]) < min(unit["end_s"], overlap["end_s"])
            ]
            identities = {
                mapping.get(unit["session"], {}).get(unit["speaker"])
                for unit in units
                if unit["speaker"] is not None
            }
            n = len(identities & set(map(str, overlap["speakers"])))
            overlaps.append(
                {
                    "id": overlap["id"],
                    "view": name,
                    "outcome": "both" if n >= 2 else "one" if n == 1 else "neither",
                    "identified": sorted(identities - {None}),
                    "identified_count": n,
                    "reference_count": len(overlap["speakers"]),
                }
            )
    return {
        "recording_id": recording["id"],
        "language": recording["language"],
        "status": run.get("status", "connected"),
        "planned_anchors": len(anchors),
        "planned_turn_opportunities": len(recording.get("turn_opportunities", [])),
        "activity_only": activity,
        "mapping": mapping,
        "views": views,
        "overlaps": overlaps,
        "reference_scope": {
            "anchor_speakers": sorted(refs),
            "annotation_speakers": sorted({str(span["speaker"]) for span in recording["spans"]}),
            "unanchored_speakers": sorted(
                {str(span["speaker"]) for span in recording["spans"]} - refs
            ),
            "limitation": "Anchor accuracy only covers listed anchor speakers; a single-reference-speaker result is not a multispeaker accuracy estimate.",
        },
        "timing": identity_timing(recording, journal, mapping, activity),
        "processing_delay": {
            "all_calls_by_send_kind": {
                kind: timing(
                    [
                        row["processing_delay_s"]
                        for row in run.get("sends", [])
                        if row.get("kind") == kind and number(row.get("processing_delay_s"))
                    ]
                )
                for kind in sorted(
                    {
                        row.get("kind", "unknown")
                        for row in run.get("sends", [])
                        if number(row.get("processing_delay_s"))
                    }
                )
            },
            "nonempty_source_probability_updates": timing(
                [
                    row["raw"]["processing_delay_s"]
                    for row in journal.events
                    if row.get("kind") == "probabilities"
                    and row.get("raw", {}).get("count", 0) > 0
                    and number(row["raw"].get("processing_delay_s"))
                ]
            ),
            "eof_probability_updates": timing(
                [
                    row["raw"]["processing_delay_s"]
                    for row in journal.events
                    if row.get("kind") == "eof_probabilities"
                    and number(row.get("raw", {}).get("processing_delay_s"))
                ]
            ),
            "limit": "Per-call model processing, including empty source updates in the all-call series; separate from the declared input buffer and source-onset identity timing. Not app-caption delay.",
        },
        "identity_structure": {
            view: [
                {
                    "reference": ref,
                    "native_ids": sorted(
                        {
                            (row["session"], row["speaker"])
                            for row in views[view]["anchors"]
                            if str(row["reference"]) == ref and row["state"] == "assigned"
                        }
                    ),
                }
                for ref in sorted(refs)
            ]
            for view in views
        },
        "changes": change_cases(recording, journal, mapping, activity),
        "turns": (
            turn_score(recording, journal)
            if run["arm"] in {"E0", "T1", "T2"} and recording.get("diagnostic")
            else None
        ),
        "late_metadata": late_metadata(journal) if run["arm"] in {"E0", "T1", "T2"} else None,
        "caption_structure": (
            caption_structure(run, recording, journal, mapping) if run["arm"] == "S1" else None
        ),
        "run": {
            key: run.get(key)
            for key in (
                "status",
                "errors",
                "sent_audio_s",
                "connection_duration_s",
                "eof",
                "drain",
                "metadata",
                "counts",
            )
        },
    }


def identity_timing(recording, journal, mapping, activity):
    arrivals = sorted(
        {row["arrival_s"] for row in journal.events}
        | {row["arrival_s"] for row in journal.run.get("observations", [])}
    )
    cases = []
    seen = set()
    for span in sorted(recording["spans"], key=lambda row: row["start_s"]):
        ref = str(span["speaker"])
        if ref in seen:
            continue
        seen.add(ref)
        found = usable = None
        reference_supported = ref in {str(anchor["speaker"]) for anchor in recording["anchors"]}
        if activity:
            candidates = sorted(journal.activity, key=lambda row: row["arrival_s"])
            states = ((row["arrival_s"], [row]) for row in candidates)
        else:
            states = (
                (arrival, journal.snapshot(arrival))
                for arrival in arrivals
                if arrival >= span["start_s"]
            )
        for arrival, units in states:
            candidates = [
                unit
                for unit in units
                if number(unit.get("start_s"))
                and number(unit.get("end_s"))
                and max(unit["start_s"], span["start_s"]) < min(unit["end_s"], span["end_s"])
                and unit.get("speaker") is not None
                and (activity or lexical(unit.get("text")))
            ]
            if candidates and usable is None:
                usable = {
                    "arrival_s": arrival,
                    "delay_s": arrival - span["start_s"],
                    "evidence": candidates[0].get("evidence"),
                }
            if usable is not None and not reference_supported:
                break
            correct = [
                unit
                for unit in candidates
                if mapping.get(unit["session"], {}).get(unit["speaker"]) == ref
            ]
            if correct:
                found = {
                    "arrival_s": arrival,
                    "delay_s": arrival - span["start_s"],
                    "evidence": correct[0].get("evidence"),
                }
                break
        cases.append(
            {
                "speaker": ref,
                "onset_s": span["start_s"],
                "status": (
                    "reference_unanchored"
                    if not reference_supported
                    else "observed" if found else "unavailable"
                ),
                "first_usable": usable,
                **(found or {}),
            }
        )
    commits = []
    for commit in journal.commits():
        index = commit["segment_index"]
        found = None
        if index is not None:
            for arrival in arrivals:
                if arrival < commit["arrival_s"]:
                    continue
                units = [
                    unit
                    for unit in journal.snapshot(arrival)
                    if unit.get("segment_index") == index and unit["session"] == commit["session"]
                ]
                if any(
                    unit.get("speaker") is not None
                    and number(unit.get("start_s"))
                    and number(unit.get("end_s"))
                    and lexical(unit.get("text"))
                    for unit in units
                ):
                    found = arrival
                    break
        commits.append(
            {
                "segment_index": index,
                "commit_arrival_s": commit["arrival_s"],
                "identity_arrival_s": found,
                "delay_s": found - commit["arrival_s"] if found is not None else None,
                "evidence": commit["evidence"],
                "status": "observed" if found is not None else "unavailable",
            }
        )
    return {
        "new_speaker_correct_identity": timing(
            [row["delay_s"] for row in cases if "delay_s" in row]
        ),
        "new_speaker_first_usable": timing(
            [row["first_usable"]["delay_s"] for row in cases if row["first_usable"]]
        ),
        "new_speaker_cases": cases,
        "commit_to_first_timed_word_identity": timing(
            [row["delay_s"] for row in commits if row["delay_s"] is not None]
        ),
        "commit_cases": commits,
    }


def span_edge(units, span, edge, activity):
    candidates = [
        unit
        for unit in units
        if number(unit.get("start_s"))
        and number(unit.get("end_s"))
        and max(unit["start_s"], span["start_s"]) < min(unit["end_s"], span["end_s"])
        and (activity or lexical(unit.get("text")))
    ]
    if not candidates:
        return {
            "state": (
                "missing_speaker"
                if activity
                else (
                    "no_timed_text_coverage"
                    if any(source_lexical_unit(unit) for unit in units)
                    else "missing_text"
                )
            )
        }
    endpoint_key = "start_s" if edge == "first" else "end_s"
    endpoint = (min if edge == "first" else max)(unit[endpoint_key] for unit in candidates)
    tied = [unit for unit in candidates if unit[endpoint_key] == endpoint]
    priority = max(
        bool(unit.get("settled")) and unit.get("segment_index") is not None for unit in tied
    )
    tied = [
        unit
        for unit in tied
        if (bool(unit.get("settled")) and unit.get("segment_index") is not None) == priority
    ]
    newest = max(unit["arrival_s"] for unit in tied)
    tied = [unit for unit in tied if unit["arrival_s"] == newest]
    identities = {(unit["session"], unit.get("speaker")) for unit in tied}
    if len(identities) != 1:
        return {"state": "missing_speaker", "detail": "conflicting_exact_edge_identities"}
    selected = tied[0]
    if selected.get("speaker") is None:
        return {"state": "missing_speaker", "evidence": selected.get("evidence")}
    return {
        "state": "assigned",
        "session": selected["session"],
        "speaker": selected["speaker"],
        "evidence": selected.get("evidence"),
        "source_interval_s": [selected["start_s"], selected["end_s"]],
        "selection": "indexed_settled_then_latest_receipt_at_exact_endpoint",
    }


def change_cases(recording, journal, mapping, activity):
    result = []
    spans = {span["id"]: span for span in recording["spans"]}
    for change in recording.get("changes", []):
        source, target = spans.get(change.get("source_span_id")), spans.get(
            change.get("target_span_id")
        )
        if source is not None and target is not None:
            cutoff = target["end_s"] + 1
            anchor_ids = None
        else:
            before = [
                anchor
                for anchor in recording["anchors"]
                if anchor["time_s"] < change["time_s"]
                and str(anchor["speaker"]) == str(change["from_speaker"])
            ]
            after = [
                anchor
                for anchor in recording["anchors"]
                if anchor["time_s"] >= change["time_s"]
                and str(anchor["speaker"]) == str(change["to_speaker"])
            ]
            if not before or not after:
                result.append(
                    {
                        "id": change["id"],
                        "status": "unavailable",
                        "reason": "no_frozen_spans_or_bracketing_anchors",
                    }
                )
                continue
            pair = [
                max(before, key=lambda row: row["time_s"]),
                min(after, key=lambda row: row["time_s"]),
            ]
            cutoff = pair[1]["span_end_s"] + 1
            anchor_ids = [anchor["id"] for anchor in pair]
        for view, limit in (("online", cutoff), ("eventual", math.inf)):
            units = journal.snapshot(limit)
            values = (
                [
                    span_edge(units, source, "last", activity),
                    span_edge(units, target, "first", activity),
                ]
                if source is not None and target is not None
                else [at_anchor(units, anchor, recording, activity) for anchor in pair]
            )
            ids = [(value.get("session"), value.get("speaker")) for value in values]
            available = all(value["state"] == "assigned" for value in values)
            refs = [str(change["from_speaker"]), str(change["to_speaker"])]
            is_change = refs[0] != refs[1]
            identity_supported = set(refs) <= {
                str(anchor["speaker"]) for anchor in recording["anchors"]
            }
            result.append(
                {
                    "id": change["id"],
                    "view": view,
                    "kind": change.get("kind"),
                    "time_s": change["time_s"],
                    "anchor_ids": anchor_ids,
                    "source_span_id": change.get("source_span_id"),
                    "target_span_id": change.get("target_span_id"),
                    "sampling": (
                        "Last/first received lexical or activity interval intersecting frozen source/target spans; separate opportunity probe, not extra fixed anchors."
                        if source is not None and target is not None
                        else "Bracketing fixed anchors"
                    ),
                    "status": "observed" if available else "unavailable",
                    "native_ids": ids,
                    "evidence": values,
                    "missed_change": available and is_change and ids[0] == ids[1],
                    "false_change": available and not is_change and ids[0] != ids[1],
                    "identity_reference_supported": identity_supported,
                    "correct_identity_pair": (
                        available
                        and all(
                            mapping.get(value["session"], {}).get(value["speaker"]) == ref
                            for value, ref in zip(values, refs, strict=True)
                        )
                        if identity_supported
                        else None
                    ),
                }
            )
    return result


def turn_score(recording, journal):
    commits = journal.commits()
    spans = {span["id"]: span for span in recording["spans"]}
    cases = []
    for opportunity in recording.get("turn_opportunities", []):
        time = opportunity["time_s"]
        start, end = opportunity.get("start_s", time), opportunity.get("end_s", time)
        selected_spans = [spans[key] for key in opportunity.get("span_ids", [])]
        left = selected_spans[0] if selected_spans else {"start_s": start, "end_s": time}
        right = selected_spans[1] if len(selected_spans) > 1 else None
        if right is None and opportunity["kind"] == "ending":
            following = [
                span
                for span in recording["spans"]
                if span["start_s"] >= time and span["id"] != left.get("id")
            ]
            right = min(following, key=lambda row: row["start_s"]) if following else None

        def covered(span):
            return [
                (commit, word)
                for commit in commits
                for word in commit["words"]
                if max(word["start_s"], span["start_s"]) < min(word["end_s"], span["end_s"])
            ]

        left_words = covered(left)
        right_words = covered(right) if right else []
        last = max(left_words, key=lambda row: row[1]["end_s"])[0] if left_words else None
        first_right = (
            min(right_words, key=lambda row: row[1]["start_s"])[0] if right_words else None
        )
        pair_available = last is not None and first_right is not None
        split = last is not first_right if pair_available else None
        cuts = [
            commit
            for commit in commits
            if number(commit["end_s"]) and start <= commit["end_s"] <= end
        ]
        nearest_candidates = [
            commit
            for commit in commits
            if number(commit["end_s"])
            and left["start_s"] <= commit["end_s"] <= (right["end_s"] if right else left["end_s"])
        ]
        nearest = (
            min(nearest_candidates, key=lambda row: abs(row["end_s"] - time))
            if nearest_candidates
            else None
        )
        timing_available = any(number(commit["end_s"]) for commit in commits)
        if len(selected_spans) > 1:
            unsafe_cut = split
            intermediate = split
        elif start < end:
            unsafe_cut = bool(cuts) if timing_available else None
            intermediate = bool(cuts) if timing_available else None
        else:
            unsafe_cut = intermediate = None
        all_words = left_words + right_words
        unique_words = {id(word): word for _, word in all_words}
        ordered_words = sorted(
            unique_words.values(), key=lambda row: (row["start_s"], row["end_s"])
        )
        text = "".join(word["text"] for word in ordered_words)
        observed = normalized(text)
        references = [
            normalized(part)
            for part in opportunity.get("reference_text", "").split("|")
            if normalized(part)
        ]
        occurrences = [
            len(
                lexical_reference_matches(
                    [normalized(word["text"]) for word in ordered_words], part
                )
            )
            for part in references
        ]
        completion = (
            ending_completion(left_words, opportunity.get("reference_text"), time)
            if opportunity["kind"] == "ending"
            else None
        )
        contexts = [
            commit.get("control_state", {}).get("last_command") or {}
            for commit in (last, first_right)
            if commit
        ]
        manual_contexts = [
            context
            for commit in (last, first_right)
            if commit
            for context in commit["possible_manual_commands"]
        ]
        excluded_context = any(
            context.get("reason") in {"cap", "eof"} for context in manual_contexts
        )
        evidence = list(dict.fromkeys(commit["evidence"] for commit, _ in all_words))
        cases.append(
            {
                "id": opportunity["id"],
                "kind": opportunity["kind"],
                "time_s": time,
                "uncertainty": opportunity.get("semantic_uncertainty", opportunity.get("notes")),
                "timing_available": timing_available,
                "pair_available": pair_available,
                "premature_cut": unsafe_cut if opportunity["kind"] == "unsafe" else None,
                "merged_boundary": (
                    not split if opportunity["kind"] == "ending" and pair_available else None
                ),
                "ending_to_commit_s": completion["delay_s"] if completion else None,
                "ending_completion": completion,
                "intermediate_output": (
                    intermediate
                    if opportunity["kind"] == "intermediate" and not excluded_context
                    else None
                ),
                "nearest_boundary_delta_s": nearest["end_s"] - time if nearest else None,
                "nearest_boundary_evidence": nearest["evidence"] if nearest else None,
                "cause": last["cause"] if last else None,
                "command_contexts_not_causal": contexts,
                "possible_manual_contexts_not_causal": manual_contexts,
                "cap_eof_context_excluded_from_useful": excluded_context,
                "content": {
                    "reference": opportunity.get("reference_text"),
                    "observed": text,
                    "exact_normalized": observed == "".join(references) if references else None,
                    "reference_contained": all(occurrences) if references else None,
                    "part_occurrences": occurrences,
                },
                "commit_evidence": evidence,
                "boundary_limit": (
                    "Single-point internal-clause hypotheses have no tolerance window; nearest signed boundary offset is reported, not a manufactured binary success/failure."
                    if len(selected_spans) == 1 and start == end and opportunity["kind"] != "ending"
                    else None
                ),
            }
        )
    delays = [row["ending_to_commit_s"] for row in cases if row["ending_to_commit_s"] is not None]
    commands = [row for row in journal.events if row.get("kind") == "commit_command"]
    long_speech = []
    for span in recording["spans"]:
        if span["end_s"] - span["start_s"] <= 10:
            continue
        boundaries = [
            commit
            for commit in commits
            if number(commit["end_s"]) and span["start_s"] < commit["end_s"] < span["end_s"]
        ]
        long_speech.append(
            {
                "span_id": span["id"],
                "duration_s": span["end_s"] - span["start_s"],
                "intermediate_commits": len(boundaries),
                "non_cap_eof_context_commits": sum(
                    not row["empty_text"]
                    and not any(
                        context.get("reason") in {"cap", "eof"}
                        for context in row["possible_manual_commands"]
                    )
                    for row in boundaries
                ),
                "boundaries": [
                    {
                        key: row.get(key)
                        for key in (
                            "end_s",
                            "arrival_s",
                            "cause",
                            "control_state",
                            "possible_manual_commands",
                            "evidence",
                        )
                    }
                    for row in boundaries
                ],
            }
        )
    return {
        "opportunities": cases,
        "commits": [
            {key: value for key, value in row.items() if key != "words"} for row in commits
        ],
        "long_speech": long_speech,
        "counts": {
            "opportunities": len(cases),
            "commits_total": len(commits),
            "fragments_translation_triggers": sum(not row["empty_text"] for row in commits),
            "empty_commits": sum(row["empty_text"] for row in commits),
            "premature_cuts": sum(row["premature_cut"] is True for row in cases),
            "unsafe_evaluable": sum(row["premature_cut"] is not None for row in cases),
            "merged_endings": sum(row["merged_boundary"] is True for row in cases),
            "endings_pair_evaluable": sum(row["merged_boundary"] is not None for row in cases),
            "useful_intermediate": sum(row["intermediate_output"] is True for row in cases),
            "intermediate_evaluable": sum(row["intermediate_output"] is not None for row in cases),
            "endings_latency_eligible": sum(row["ending_to_commit_s"] is not None for row in cases),
            "endings_latency_unavailable": sum(
                row["kind"] == "ending" and row["ending_to_commit_s"] is None for row in cases
            ),
            "timing_unavailable_commits": sum(not number(row["end_s"]) for row in commits),
            "causes": dict(Counter(row["cause"] for row in commits)),
            "commands_not_causal": dict(Counter(row["raw"].get("reason") for row in commands)),
        },
        "ending_to_commit": timing(delays),
        "limits": "Annotation-derived opportunities; signed delays can expose annotation/model timing disagreement. Content containment is a mismatch probe, not semantic adjudication. Paired cuts use last left/first right recognized words; missing/unindexed evidence is unavailable. Cap/EOF command context is not native commit cause or successful prediction. First/last commits and all command contexts remain in per-commit evidence for startup/EOF separation.",
    }


def late_metadata(journal):
    cases = []
    for commit in journal.commits():
        index = commit["segment_index"]
        if index is None:
            continue
        before = [
            unit
            for unit in journal.snapshot(commit["arrival_s"])
            if unit.get("segment_index") == index and unit["session"] == commit["session"]
        ]
        after = [
            unit
            for unit in journal.snapshot()
            if unit.get("segment_index") == index and unit["session"] == commit["session"]
        ]
        earlier = current_composition_projection(before)
        later = current_composition_projection(after)
        before_text = [group["normalized_source_text"] for group in earlier]
        after_text = [group["normalized_source_text"] for group in later]
        changed = [(group["scope"], group["representative"]) for group in earlier] != [
            (group["scope"], group["representative"]) for group in later
        ]
        text_changed = "".join(before_text) != "".join(after_text)
        cases.append(
            {
                "segment_index": index,
                "evidence": commit["evidence"],
                "speaker_changed": changed,
                "consequence": (
                    "source_text_changed"
                    if text_changed
                    else (
                        "regroup_retranslate_required"
                        if before_text != after_text
                        else "identity_only" if changed else "none"
                    )
                ),
                "before_source_groups": earlier,
                "eventual_source_groups": later,
                "text_changed": text_changed,
                "grouping_changed": before_text != after_text if not text_changed else None,
            }
        )
    return {
        "projection": "counterfactual_current_composition",
        "counts": dict(Counter(row["consequence"] for row in cases)),
        "cases": cases,
        "limit": "Identity-neutral counterfactual projection of current #373 unknown attachment onto source text (NFKC and whitespace-only normalization, punctuation/case preserved). Text changes are separate from comparable source partitions. Real Scribe payloads are never labeled Soniox; unobserved language boundaries cannot be evaluated. This is not actual Scribe late-repair/MT/caption integration, and no translated string is split.",
    }


def caption_structure(run, recording, journal, mapping):
    exports = run.get("application", {})
    if not exports:
        return {
            "status": "unavailable",
            "reason": "same_session_group_caption_request_exports_missing",
        }
    origin = run["source_origin_mono_s"]
    scopes = {
        row["speaker_session_scope"]: str(row["session_id"])
        for row in run.get("session_scopes", [])
    }
    groups, flows = [], {}
    for index, group in enumerate(exports.get("groups", [])):
        caption = group.get("accepted_caption")
        if caption is None:
            continue
        target = caption["utterance_id"]
        previous = group.get("previous")
        flows[target] = (
            flows.get(previous, previous)
            if group.get("kind") == "continuation" and previous
            else flows.get(target, target)
        )
        transcript = group["transcript"]
        units = []
        for native in transcript.get("final_speaker_runs", []):
            start, end = native.get("source_start_ms"), native.get("source_end_ms")
            session = scopes.get(native.get("session_scope"))
            unit = {
                "start_s": start / 1000 if number(start) else None,
                "end_s": end / 1000 if number(end) else None,
                "text": native.get("text", ""),
                "speaker": speaker(native.get("speaker_id")),
                "session": session or "unmapped_scope",
                "arrival_s": group["t"] - origin,
                "settled": True,
                "evidence": f"{journal.evidence}#application/groups/{index}",
            }
            units.extend(
                to_source(unit, run)
                if session is not None
                else [{**unit, "start_s": None, "end_s": None}]
            )
        if not units and lexical(transcript.get("text")):
            units = [
                {
                    "start_s": None,
                    "end_s": None,
                    "text": transcript["text"],
                    "speaker": None,
                    "session": "unmapped_scope",
                    "arrival_s": group["t"] - origin,
                    "settled": True,
                }
            ]
        groups.append(
            {
                "target": target,
                "flow": flows[target],
                "arrival_s": group["t"] - origin,
                "units": units,
                "kind": group.get("kind"),
                "text": transcript["text"],
                "evidence": f"{journal.evidence}#application/groups/{index}",
            }
        )
    events = exports.get("events", [])
    views = {}
    for view in ("online", "eventual"):
        anchors = []
        for anchor in recording["anchors"]:
            cutoff = anchor["span_end_s"] + 1 if view == "online" else math.inf
            latest = {}
            for row in events:
                if row["t"] - origin <= cutoff:
                    event = row["event"]
                    latest[event["utterance_id"]] = row
            units, targets = [], []
            for group in groups:
                output = latest.get(group["target"])
                if group["arrival_s"] > cutoff or output is None:
                    continue
                event = output["event"]
                assignment = event.get("speaker_assignment") or {}
                key = (assignment.get("attribution") or {}).get("key") or {}
                identity = speaker(key.get("local_speaker_id"))
                session = scopes.get(key.get("speaker_scope_id"), "unmapped_scope")
                for unit in group["units"]:
                    units.append(
                        {
                            **unit,
                            "speaker": identity,
                            "session": session,
                            "arrival_s": output["t"] - origin,
                            "settled": bool(event.get("is_final")),
                            "evidence": group["evidence"],
                        }
                    )
                    if (
                        number(unit.get("start_s"))
                        and number(unit.get("end_s"))
                        and unit["start_s"] <= anchor["time_s"] < unit["end_s"]
                    ):
                        targets.append(
                            {
                                "target": group["target"],
                                "flow": group["flow"],
                                "evidence": group["evidence"],
                            }
                        )
            value = at_anchor(units, anchor, recording)
            outcome = value["state"]
            if outcome == "assigned":
                outcome = (
                    "correct"
                    if mapping.get(value["session"], {}).get(value["speaker"])
                    == str(anchor["speaker"])
                    else "wrong"
                )
            anchors.append(
                {
                    "anchor_id": anchor["id"],
                    "reference": anchor["speaker"],
                    "outcome": outcome,
                    "targets": targets,
                    **value,
                }
            )
        views[view] = {"counts": counts(anchors), "anchors": anchors}
    by_anchor = {row["anchor_id"]: row for row in views["eventual"]["anchors"]}
    composition = []
    for change in change_cases(recording, journal, mapping, False):
        if change.get("view") != "eventual":
            continue
        if change.get("anchor_ids"):
            left, right = [by_anchor[key] for key in change["anchor_ids"]]
            a = {row["flow"] for row in left["targets"]}
            b = {row["flow"] for row in right["targets"]}
            different = left["reference"] != right["reference"]
        else:
            spans = {span["id"]: span for span in recording["spans"]}
            left, right = spans[change["source_span_id"]], spans[change["target_span_id"]]

            def edge_flow(span, edge):
                candidates = [
                    (group, unit)
                    for group in groups
                    for unit in group["units"]
                    if number(unit.get("start_s"))
                    and number(unit.get("end_s"))
                    and max(unit["start_s"], span["start_s"]) < min(unit["end_s"], span["end_s"])
                ]
                if not candidates:
                    return set()
                chosen = (
                    min(candidates, key=lambda row: row[1]["start_s"])
                    if edge == "first"
                    else max(candidates, key=lambda row: row[1]["end_s"])
                )
                return {chosen[0]["flow"]}

            a, b = edge_flow(left, "last"), edge_flow(right, "first")
            different = left["speaker"] != right["speaker"]
        composition.append(
            {
                "change_id": change["id"],
                "anchor_ids": change.get("anchor_ids"),
                "source_span_id": change.get("source_span_id"),
                "target_span_id": change.get("target_span_id"),
                "available": bool(a and b),
                "same_flow_at_selected_envelopes": bool(a & b),
                "cross_speaker_envelope_candidate": bool(a & b) and different,
                "native_edge_same_id_probe": change["missed_change"],
            }
        )
    fragmentation = []
    for span in recording["spans"]:
        anchor_ids = [
            anchor["id"] for anchor in recording["anchors"] if anchor["span_id"] == span["id"]
        ]
        active_flows = {
            target["flow"] for key in anchor_ids for target in by_anchor[key]["targets"]
        }
        if anchor_ids:
            fragmentation.append(
                {
                    "span_id": span["id"],
                    "flows": sorted(active_flows),
                    "unnecessary_fragmentation_probe": len(active_flows) > 1,
                }
            )
    finals = [row for row in events if row["event"].get("is_final")]
    revisions = Counter(row["event"]["utterance_id"] for row in finals)
    return {
        "status": "available",
        "mode": exports.get("mode"),
        "views": views,
        "counts": {
            "groups": len(groups),
            "requests": len(exports.get("requests", [])),
            "caption_preparations": len(exports.get("captions", [])),
            "final_outputs": len(finals),
            "targets_with_final_revisions": sum(value > 1 for value in revisions.values()),
            "cross_speaker_envelope_candidates": sum(
                row["cross_speaker_envelope_candidate"] for row in composition
            ),
            "fragmented_reference_span_probes": sum(
                row["unnecessary_fragmentation_probe"] for row in fragmentation
            ),
        },
        "composition": composition,
        "fragmentation": fragmentation,
        "groups": [{key: value for key, value in row.items() if key != "units"} for row in groups],
        "limit": "Envelope candidate probes, not established harmful joins or provider accuracy. Source-run bounds can spill across reference boundaries or fill gaps between native tokens; corroborate raw group/request/caption IDs before attributing harm. Broad-envelope assigned coverage is not recovered ASR text. Structural controlled-MT replay is not actual MT/GUI latency; RGB is not speaker truth.",
    }


def aggregate(results):
    available = [result for result in results if "views" in result]
    summary = {
        "connected_recordings": len(available),
        "unrun_recordings": len(results) - len(available),
        "planned_anchors": sum(row["planned_anchors"] for row in results),
        "planned_turn_opportunities": sum(row["planned_turn_opportunities"] for row in results),
    }
    for view in ("online", "eventual"):
        rows = [row for result in available for row in result["views"][view]["anchors"]]
        summary[view] = {
            "all": counts(rows),
            "by_language": {
                language: counts(
                    [
                        row
                        for result in available
                        if result["language"] == language
                        for row in result["views"][view]["anchors"]
                    ]
                )
                for language in sorted({result["language"] for result in results})
            },
        }
    onset = [row for result in available for row in result["timing"]["new_speaker_cases"]]
    commits = [row for result in available for row in result["timing"]["commit_cases"]]
    summary["timing"] = {
        "new_speaker_correct_identity": timing(
            [row["delay_s"] for row in onset if row.get("delay_s") is not None]
        ),
        "new_speaker_first_usable": timing(
            [row["first_usable"]["delay_s"] for row in onset if row["first_usable"]]
        ),
        "new_speaker_unavailable": sum(row["status"] == "unavailable" for row in onset),
        "new_speaker_reference_unanchored": sum(
            row["status"] == "reference_unanchored" for row in onset
        ),
        "commit_to_first_timed_word_identity": timing(
            [row["delay_s"] for row in commits if row.get("delay_s") is not None]
        ),
        "commit_identity_unavailable": sum(row["status"] == "unavailable" for row in commits),
    }
    turns = [result["turns"] for result in available if result.get("turns")]
    summary["turns"] = {
        "counts": {
            key: sum(turn["counts"][key] for turn in turns)
            for key in (
                "opportunities",
                "commits_total",
                "fragments_translation_triggers",
                "empty_commits",
                "premature_cuts",
                "unsafe_evaluable",
                "merged_endings",
                "endings_pair_evaluable",
                "useful_intermediate",
                "intermediate_evaluable",
                "timing_unavailable_commits",
                "endings_latency_eligible",
                "endings_latency_unavailable",
            )
        },
        "ending_to_commit": timing(
            [
                row["ending_to_commit_s"]
                for turn in turns
                for row in turn["opportunities"]
                if row["ending_to_commit_s"] is not None
            ]
        ),
        "commands_not_causal": dict(
            sum((Counter(turn["counts"]["commands_not_causal"]) for turn in turns), Counter())
        ),
    }
    return summary


def comparisons(arms):
    results = []
    s0 = {row["recording_id"]: row for row in arms["S0"]["recordings"]}
    for native in arms["S1"]["recordings"]:
        recording_id = native["recording_id"]
        for label, left, right in (
            ("S0_vs_S1_native_input_context_finalization", s0[recording_id], native),
            ("S1_native_vs_S1_caption_composition", native, native.get("caption_structure") or {}),
        ):
            if "views" not in left or "views" not in right:
                results.append(
                    {"comparison": label, "recording_id": recording_id, "status": "unavailable"}
                )
                continue
            for view in ("online", "eventual"):
                a, b = left["views"][view]["anchors"], right["views"][view]["anchors"]
                if [row["anchor_id"] for row in a] != [row["anchor_id"] for row in b]:
                    raise ValueError("Paired comparison changed fixed anchors")
                pairs = Counter((x["outcome"], y["outcome"]) for x, y in zip(a, b, strict=True))
                results.append(
                    {
                        "comparison": label,
                        "recording_id": recording_id,
                        "view": view,
                        "status": "observed",
                        "eligible": len(a),
                        "outcome_pairs": [
                            {"left": x, "right": y, "n": n} for (x, y), n in sorted(pairs.items())
                        ],
                    }
                )
    return results


def representative_cases(arms):
    candidates = []
    for arm, value in arms.items():
        for result in value["recordings"]:
            if "views" not in result:
                continue
            prefix = {"arm": arm, "recording_id": result["recording_id"]}
            for online, eventual in zip(
                result["views"]["online"]["anchors"],
                result["views"]["eventual"]["anchors"],
                strict=True,
            ):
                if online["outcome"] != eventual["outcome"] and (
                    eventual.get("evidence") or online.get("evidence")
                ):
                    candidates.append(
                        {
                            **prefix,
                            "kind": "online_eventual_change",
                            "online": online,
                            "eventual": eventual,
                        }
                    )
            observed = [
                row
                for row in result["timing"]["new_speaker_cases"]
                if row.get("delay_s") is not None
            ]
            if observed:
                candidates.append(
                    {
                        **prefix,
                        "kind": "slow_new_speaker",
                        "case": max(observed, key=lambda row: row["delay_s"]),
                    }
                )
            turns = result.get("turns") or {}
            for case in turns.get("opportunities", []):
                if case.get("premature_cut") or case.get("merged_boundary"):
                    candidates.append({**prefix, "kind": "turn_boundary", "case": case})
            metadata = result.get("late_metadata") or {}
            for case in metadata.get("cases", []):
                if case["consequence"] == "regroup_retranslate_required":
                    candidates.append({**prefix, "kind": "late_regrouping", "case": case})
            caption = result.get("caption_structure") or {}
            for case in caption.get("composition", []):
                if case["cross_speaker_envelope_candidate"]:
                    candidates.append(
                        {**prefix, "kind": "caption_envelope_candidate", "case": case}
                    )
    selected = []
    for candidate in candidates:
        if candidate["kind"] not in {row["kind"] for row in selected}:
            selected.append(candidate)
    for candidate in candidates:
        if len(selected) >= 5:
            break
        if candidate not in selected:
            selected.append(candidate)
    return selected[:5]


def score(manifest, runs_root):
    if manifest.get("schema_version") != 1 or not manifest.get("frozen_at"):
        raise ValueError("Scoring requires schema_version=1 and a frozen manifest")
    arms = {}
    for arm in ARMS:
        results = []
        for recording in manifest["recordings"]:
            if arm in {"T1", "T2"} and not recording.get("diagnostic"):
                continue
            path = Path(runs_root) / arm / f"{recording['id']}.json"
            run = load(path) if path.exists() else None
            if run and (run.get("arm") != arm or run.get("recording_id") != recording["id"]):
                raise ValueError(f"Run identity mismatch: {path}")
            results.append(score_recording(recording, run, path.as_posix()))
        arms[arm] = {"summary": aggregate(results), "recordings": results}
        for result in results:
            path = Path(runs_root) / arm / f"{result['recording_id']}.json"
            result["evidence_path"] = path.as_posix()
            result["log_sha256"] = (
                hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            )
    return {
        "schema_version": 1,
        "reference_method": manifest.get("reference_method"),
        "frozen_at": manifest["frozen_at"],
        "mapping_method": MAPPING_METHOD,
        "anchor_policy": "All frozen anchors remain eligible in connected sessions, including missing text/timing/identity. N is activity-only. Unconnected sessions are unrun, not model failures.",
        "unassigned_reason_policy": {
            "missing_text": "No lexical text evidence received anywhere in this session view; not an anchor-local lexical-recall estimate.",
            "missing_timing": "Untimed or unusably timed lexical evidence has an explicit source range containing this anchor.",
            "missing_speaker": "Timed text covers the anchor but identity is absent/conflicting; N instead has no usable activity identity.",
            "no_timed_text_coverage": "Text exists, but no usable timed lexical unit covers the fixed anchor and the local cause is unsupported. This does not establish missing recognition or justify interpolation.",
        },
        "arms": arms,
        "comparisons": comparisons(arms),
        "representative_cases": representative_cases(arms),
        "T0": "E0 diagnostic recordings reused, not independently executed",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=Path("benchmarks/asr/issue441-inputs.json"))
    parser.add_argument("--runs", type=Path, default=Path(".tmp/issue441/runs"))
    parser.add_argument("--output", type=Path, default=Path(".tmp/issue441/scores/score.json"))
    args = parser.parse_args()
    result = score(load(args.inputs), args.runs)
    result["manifest_sha256"] = hashlib.sha256(args.inputs.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for arm, value in result["arms"].items():
        summary = value["summary"]
        print(
            arm,
            json.dumps(
                {
                    key: summary[key]
                    for key in ("connected_recordings", "unrun_recordings", "online", "eventual")
                },
                ensure_ascii=False,
            ),
        )


if __name__ == "__main__":
    main()
