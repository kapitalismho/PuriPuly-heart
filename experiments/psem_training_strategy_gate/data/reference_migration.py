from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.psem_training_strategy_gate.data.alimeeting_train_selection import (
    validate_selection_receipt,
)
from experiments.psem_training_strategy_gate.data.annotation_normalization import (
    NormalizedSession,
    normalize_inventory,
)
from experiments.psem_training_strategy_gate.data.forced_alignment_reference import (
    REFERENCE_COMMIT,
    REFERENCE_REPOSITORY,
)
from experiments.psem_training_strategy_gate.data.label_contract import (
    CONTRACT_PATH_BY_VERSION,
    CanonicalInterval,
    LabelResult,
    generate_labels,
    load_contract,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    EXPECTED_ALIMEETING_MEETINGS,
    EXPECTED_AMI_MEETINGS,
    canonical_sha256,
    sha256_file,
    write_jsonl,
)
from experiments.psem_training_strategy_gate.data.reference_normalization import (
    EXPECTED_INVENTORY_SHA256,
    INVENTORY_PATH,
    MASK_CLASS,
    ReferenceNormalizedSession,
    normalize_reference_inventory,
)

SAMPLE_RATE_HZ = 16000
MATCH_THRESHOLDS_MS = (50, 100, 200, 500)
MATCH_LIMIT_SAMPLES = MATCH_THRESHOLDS_MS[-1] * SAMPLE_RATE_HZ // 1000
DIRECT_TOPOLOGIES = frozenset({"clean_direct_different_speaker_handoff"})
GAP_TOPOLOGIES = frozenset(
    {"silence_gap_different_speaker_handoff", "same_speaker_silence_gap_resume"}
)
OVERLAP_TOPOLOGIES = frozenset({"overlap_takeover", "overlap_return"})
TOPOLOGY_FAMILIES = ("direct", "gap", "overlap")
OFFICIAL_TOPOLOGIES = (
    "short_backchannel_return",
    "overlap_takeover",
    "overlap_return",
    "silence_gap_different_speaker_handoff",
    "same_speaker_silence_gap_resume",
    "clean_direct_different_speaker_handoff",
)


class ReferenceMigrationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    speaker_id: str
    start_sample: int
    end_sample: int


@dataclass(frozen=True, slots=True)
class TopologyEvent:
    identity: tuple[str, ...]
    topology: str
    start_sample: int
    end_sample: int


@dataclass(frozen=True, slots=True)
class SessionAnalysis:
    row: dict[str, Any]
    start_displacements: tuple[int, ...]
    end_displacements: tuple[int, ...]
    absolute_displacements: tuple[int, ...]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceMigrationError(f"invalid JSONL artifact: {path.name}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ReferenceMigrationError(f"JSONL artifact must contain objects: {path.name}")
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _hours(samples: int) -> float:
    return round(samples / SAMPLE_RATE_HZ / 3600, 9)


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 9) if denominator else 0.0


def _quantile(values: Sequence[int], numerator: int, denominator: int) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    scaled = (len(ordered) - 1) * numerator
    lower = scaled // denominator
    remainder = scaled % denominator
    if not remainder:
        return ordered[lower]
    upper = lower + 1
    return (ordered[lower] * (denominator - remainder) + ordered[upper] * remainder) // denominator


def _quantiles(values: Sequence[int]) -> dict[str, int | None]:
    return {
        "p00": _quantile(values, 0, 100),
        "p25": _quantile(values, 25, 100),
        "p50": _quantile(values, 50, 100),
        "p75": _quantile(values, 75, 100),
        "p90": _quantile(values, 90, 100),
        "p95": _quantile(values, 95, 100),
        "p99": _quantile(values, 99, 100),
        "p100": _quantile(values, 100, 100),
    }


def _exposure(intervals: Sequence[CanonicalInterval], labels: LabelResult) -> dict[str, Any]:
    speech = sum(interval.duration_samples for interval in intervals if interval.active_speakers)
    silence = sum(
        interval.duration_samples for interval in intervals if not interval.active_speakers
    )
    overlap = sum(
        interval.duration_samples for interval in intervals if len(interval.active_speakers) >= 2
    )
    scored = sum(interval.duration_samples for interval in intervals)
    if speech + silence != scored or scored != labels.exposure["scored_samples"]:
        raise ReferenceMigrationError("activity exposure does not cover the scored timeline")
    reliable = labels.exposure["stable_singleton_samples"]
    return {
        "scored_samples": scored,
        "scored_hours": _hours(scored),
        "speech_samples": speech,
        "speech_hours": _hours(speech),
        "silence_samples": silence,
        "silence_hours": _hours(silence),
        "overlap_samples": overlap,
        "overlap_hours": _hours(overlap),
        "reliable_solo_samples": reliable,
        "reliable_solo_hours": _hours(reliable),
    }


def _speaker_segments(intervals: Sequence[CanonicalInterval]) -> tuple[SpeechSegment, ...]:
    by_speaker: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for interval in intervals:
        for speaker_id in interval.active_speakers:
            runs = by_speaker[speaker_id]
            if runs and runs[-1][1] == interval.start_sample:
                runs[-1] = (runs[-1][0], interval.end_sample)
            else:
                runs.append((interval.start_sample, interval.end_sample))
    return tuple(
        SpeechSegment(speaker_id, start, end)
        for speaker_id in sorted(by_speaker)
        for start, end in by_speaker[speaker_id]
    )


def _overlap_samples(left: SpeechSegment, right: SpeechSegment) -> int:
    return max(
        0, min(left.end_sample, right.end_sample) - max(left.start_sample, right.start_sample)
    )


def _subtract_coverage(
    start_sample: int, end_sample: int, segments: Sequence[SpeechSegment]
) -> tuple[tuple[int, int], ...]:
    cursor = start_sample
    uncovered: list[tuple[int, int]] = []
    for segment in sorted(segments, key=lambda row: (row.start_sample, row.end_sample)):
        start = max(start_sample, segment.start_sample)
        end = min(end_sample, segment.end_sample)
        if end <= start:
            continue
        if cursor < start:
            uncovered.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < end_sample:
        uncovered.append((cursor, end_sample))
    return tuple(uncovered)


def _speech_correspondence(
    old_intervals: Sequence[CanonicalInterval], new_intervals: Sequence[CanonicalInterval]
) -> tuple[dict[str, Any], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    old_segments = _speaker_segments(old_intervals)
    new_segments = _speaker_segments(new_intervals)
    old_links: dict[int, list[int]] = defaultdict(list)
    new_links: dict[int, list[int]] = defaultdict(list)
    for old_index, old in enumerate(old_segments):
        for new_index, new in enumerate(new_segments):
            if old.speaker_id == new.speaker_id and _overlap_samples(old, new):
                old_links[old_index].append(new_index)
                new_links[new_index].append(old_index)
    start_displacements: list[int] = []
    end_displacements: list[int] = []
    absolute_displacements: list[int] = []
    internal_removed = 0
    outer_removed = 0
    derived_old_samples = 0
    deterministic_old_segments = 0
    unpaired_old_segments = 0
    ambiguous_old_segments = 0
    for old_index, old in enumerate(old_segments):
        links = old_links.get(old_index, [])
        if not links:
            unpaired_old_segments += 1
            continue
        if any(len(new_links[new_index]) != 1 for new_index in links):
            ambiguous_old_segments += 1
            continue
        matched = [new_segments[new_index] for new_index in links]
        deterministic_old_segments += 1
        derived_old_samples += old.end_sample - old.start_sample
        first = min(matched, key=lambda row: (row.start_sample, row.end_sample))
        last = max(matched, key=lambda row: (row.end_sample, row.start_sample))
        start_delta = first.start_sample - old.start_sample
        end_delta = last.end_sample - old.end_sample
        start_displacements.append(start_delta)
        end_displacements.append(end_delta)
        absolute_displacements.extend((abs(start_delta), abs(end_delta)))
        hull_start = max(old.start_sample, first.start_sample)
        hull_end = min(old.end_sample, last.end_sample)
        for removed_start, removed_end in _subtract_coverage(
            old.start_sample, old.end_sample, matched
        ):
            duration = removed_end - removed_start
            if removed_end <= hull_start or removed_start >= hull_end:
                outer_removed += duration
            else:
                internal_removed += duration
    unpaired_new_segments = sum(1 for index in range(len(new_segments)) if not new_links.get(index))
    return (
        {
            "old_speech_segment_count": len(old_segments),
            "new_speech_segment_count": len(new_segments),
            "deterministic_old_segment_count": deterministic_old_segments,
            "ambiguous_old_segment_count": ambiguous_old_segments,
            "unpaired_old_segment_count": unpaired_old_segments,
            "unpaired_new_segment_count": unpaired_new_segments,
            "derived_old_speech_samples": derived_old_samples,
            "removed_internal_pause_samples": internal_removed,
            "removed_internal_pause_hours": _hours(internal_removed),
            "removed_outer_padding_samples": outer_removed,
            "removed_outer_padding_hours": _hours(outer_removed),
            "start_displacement_samples": _quantiles(start_displacements),
            "end_displacement_samples": _quantiles(end_displacements),
            "absolute_boundary_displacement_samples": _quantiles(absolute_displacements),
        },
        tuple(start_displacements),
        tuple(end_displacements),
        tuple(absolute_displacements),
    )


def _transition_map(labels: LabelResult) -> dict[str, Mapping[str, Any]]:
    rows = {row["transition_id"]: row for row in labels.transitions}
    if len(rows) != len(labels.transitions):
        raise ReferenceMigrationError("transition identities are duplicated")
    return rows


def _topology_events(labels: LabelResult) -> tuple[TopologyEvent, ...]:
    transitions = _transition_map(labels)
    events: list[TopologyEvent] = []
    for episode in labels.topology_episodes:
        topology = episode["primary_topology"]
        if episode["coverage_gate_eligible"] is not True or topology not in OFFICIAL_TOPOLOGIES:
            continue
        members = [transitions[transition_id] for transition_id in episode["transition_ids"]]
        if topology == "short_backchannel_return":
            if len(members) != 2:
                raise ReferenceMigrationError("short backchannel must contain two transitions")
            identity = (
                members[0]["from_speaker"],
                members[0]["to_speaker"],
                members[1]["to_speaker"],
            )
        else:
            if len(members) != 1:
                raise ReferenceMigrationError("ordinary topology must contain one transition")
            identity = (members[0]["from_speaker"], members[0]["to_speaker"])
        if any(not isinstance(value, str) or not value for value in identity):
            raise ReferenceMigrationError("coverage topology has an invalid speaker identity")
        start_sample = episode["start_sample"]
        end_sample = episode["end_sample"]
        if not isinstance(start_sample, int) or not isinstance(end_sample, int):
            if len(members) != 1 or not isinstance(members[0]["to_interval_index"], int):
                raise ReferenceMigrationError("coverage topology has no deterministic event time")
            start_sample = labels.intervals[members[0]["to_interval_index"]].start_sample
            end_sample = start_sample
        events.append(
            TopologyEvent(
                identity=identity,
                topology=topology,
                start_sample=start_sample,
                end_sample=end_sample,
            )
        )
    return tuple(
        sorted(
            events, key=lambda row: (row.start_sample, row.end_sample, row.identity, row.topology)
        )
    )


def _event_distance(left: TopologyEvent, right: TopologyEvent) -> int:
    return max(abs(left.start_sample - right.start_sample), abs(left.end_sample - right.end_sample))


def _match_identity_events(
    old_events: Sequence[TopologyEvent], new_events: Sequence[TopologyEvent]
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...], tuple[int, ...]]:
    old_by_identity: dict[tuple[str, ...], list[int]] = defaultdict(list)
    new_by_identity: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, event in enumerate(old_events):
        old_by_identity[event.identity].append(index)
    for index, event in enumerate(new_events):
        new_by_identity[event.identity].append(index)
    pairs: list[tuple[int, int]] = []
    for identity in sorted(set(old_by_identity) | set(new_by_identity)):
        old_indices = old_by_identity.get(identity, [])
        new_indices = new_by_identity.get(identity, [])
        old_count = len(old_indices)
        new_count = len(new_indices)
        score: list[list[tuple[int, int]]] = [
            [(0, 0) for _ in range(new_count + 1)] for _ in range(old_count + 1)
        ]
        action: list[list[str]] = [["" for _ in range(new_count + 1)] for _ in range(old_count + 1)]
        for old_pos in range(old_count - 1, -1, -1):
            for new_pos in range(new_count - 1, -1, -1):
                choices = [
                    (score[old_pos + 1][new_pos], 1, "old"),
                    (score[old_pos][new_pos + 1], 2, "new"),
                ]
                old_index = old_indices[old_pos]
                new_index = new_indices[new_pos]
                distance = _event_distance(old_events[old_index], new_events[new_index])
                if distance <= MATCH_LIMIT_SAMPLES:
                    tail = score[old_pos + 1][new_pos + 1]
                    choices.append(((tail[0] + 1, tail[1] - distance), 0, "match"))
                best = max(choices, key=lambda item: (item[0][0], item[0][1], -item[1]))
                score[old_pos][new_pos] = best[0]
                action[old_pos][new_pos] = best[2]
        old_pos = 0
        new_pos = 0
        while old_pos < old_count and new_pos < new_count:
            selected = action[old_pos][new_pos]
            if selected == "match":
                pairs.append((old_indices[old_pos], new_indices[new_pos]))
                old_pos += 1
                new_pos += 1
            elif selected == "old":
                old_pos += 1
            else:
                new_pos += 1
    matched_old = {old_index for old_index, _ in pairs}
    matched_new = {new_index for _, new_index in pairs}
    return (
        tuple(sorted(pairs)),
        tuple(index for index in range(len(old_events)) if index not in matched_old),
        tuple(index for index in range(len(new_events)) if index not in matched_new),
    )


def _topology_family(topology: str) -> str:
    if topology in DIRECT_TOPOLOGIES:
        return "direct"
    if topology in GAP_TOPOLOGIES:
        return "gap"
    if topology in OVERLAP_TOPOLOGIES:
        return "overlap"
    if topology == "short_backchannel_return":
        return "short_backchannel"
    raise ReferenceMigrationError(f"unsupported official topology: {topology}")


def _topology_comparison(old_labels: LabelResult, new_labels: LabelResult) -> dict[str, Any]:
    old_events = _topology_events(old_labels)
    new_events = _topology_events(new_labels)
    pairs, removed_indices, added_indices = _match_identity_events(old_events, new_events)
    exact_confusion: Counter[str] = Counter()
    family_confusion: Counter[str] = Counter()
    distances: list[int] = []
    unchanged = 0
    timing_only = 0
    topology_changed = 0
    for old_index, new_index in pairs:
        old = old_events[old_index]
        new = new_events[new_index]
        distance = _event_distance(old, new)
        distances.append(distance)
        exact_confusion[f"{old.topology}->{new.topology}"] += 1
        old_family = _topology_family(old.topology)
        new_family = _topology_family(new.topology)
        if old_family in {"direct", "gap", "overlap"} and new_family in {
            "direct",
            "gap",
            "overlap",
        }:
            family_confusion[f"{old_family}->{new_family}"] += 1
        if old.topology != new.topology:
            topology_changed += 1
        elif distance:
            timing_only += 1
        else:
            unchanged += 1
    retention = {
        f"within_{threshold_ms}ms": {
            "count": sum(
                distance <= threshold_ms * SAMPLE_RATE_HZ // 1000 for distance in distances
            ),
            "proportion_of_v1": _fraction(
                sum(distance <= threshold_ms * SAMPLE_RATE_HZ // 1000 for distance in distances),
                len(old_events),
            ),
        }
        for threshold_ms in MATCH_THRESHOLDS_MS
    }
    old_counts = Counter(event.topology for event in old_events)
    new_counts = Counter(event.topology for event in new_events)
    return {
        "v1_episode_count": len(old_events),
        "v2_episode_count": len(new_events),
        "matched_identity_within_500ms_count": len(pairs),
        "removed_episode_count": len(removed_indices),
        "added_episode_count": len(added_indices),
        "unchanged_episode_count": unchanged,
        "timing_only_change_count": timing_only,
        "topology_changing_match_count": topology_changed,
        "topology_changing_total_count": topology_changed
        + len(removed_indices)
        + len(added_indices),
        "event_displacement_samples": _quantiles(distances),
        "retention": retention,
        "exact_confusion": dict(sorted(exact_confusion.items())),
        "direct_gap_overlap_confusion": {
            f"{old_family}->{new_family}": family_confusion[
                f"{old_family}->{new_family}"
            ]
            for old_family in TOPOLOGY_FAMILIES
            for new_family in TOPOLOGY_FAMILIES
        },
        "v1_counts": {name: old_counts[name] for name in OFFICIAL_TOPOLOGIES},
        "v2_counts": {name: new_counts[name] for name in OFFICIAL_TOPOLOGIES},
        "overlap_takeover_return_changes": {
            name: new_counts[name] - old_counts[name]
            for name in ("overlap_takeover", "overlap_return")
        },
        "short_backchannel_change": new_counts["short_backchannel_return"]
        - old_counts["short_backchannel_return"],
    }


def _handoff_events(labels: LabelResult) -> tuple[TopologyEvent, ...]:
    events = []
    for row in labels.transitions:
        if (
            row["handoff_confirmed"] == 1
            and isinstance(row["from_speaker"], str)
            and isinstance(row["to_speaker"], str)
            and isinstance(row["handoff_source_sample"], int)
        ):
            events.append(
                TopologyEvent(
                    identity=(row["from_speaker"], row["to_speaker"]),
                    topology=row["primary_topology"],
                    start_sample=row["handoff_source_sample"],
                    end_sample=row["handoff_source_sample"],
                )
            )
    return tuple(
        sorted(events, key=lambda event: (event.start_sample, event.identity, event.topology))
    )


def _handoff_comparison(old_labels: LabelResult, new_labels: LabelResult) -> dict[str, Any]:
    old_events = _handoff_events(old_labels)
    new_events = _handoff_events(new_labels)
    pairs, removed, added = _match_identity_events(old_events, new_events)
    distances = [_event_distance(old_events[left], new_events[right]) for left, right in pairs]
    return {
        "v1_handoff_count": len(old_events),
        "v2_handoff_count": len(new_events),
        "matched_identity_within_500ms_count": len(pairs),
        "removed_handoff_count": len(removed),
        "added_handoff_count": len(added),
        "event_displacement_samples": _quantiles(distances),
    }


def _mask_comparison(old_labels: LabelResult, new_labels: LabelResult) -> dict[str, Any]:
    old_reasons = Counter(
        row["primary_topology"]
        for row in old_labels.transitions
        if row["mask_state"] == "masked" and not row["transition_id"].startswith("D")
    )
    new_reasons = Counter(
        row["primary_topology"]
        for row in new_labels.transitions
        if row["mask_state"] == "masked" and not row["transition_id"].startswith("D")
    )
    return {
        "v1_masked_transition_count": sum(old_reasons.values()),
        "v2_masked_transition_count": sum(new_reasons.values()),
        "count_change": sum(new_reasons.values()) - sum(old_reasons.values()),
        "v1_reasons": dict(sorted(old_reasons.items())),
        "v2_reasons": dict(sorted(new_reasons.items())),
        "reason_count_changes": {
            reason: new_reasons[reason] - old_reasons[reason]
            for reason in sorted(set(old_reasons) | set(new_reasons))
        },
    }


def analyze_session(old: NormalizedSession, new: ReferenceNormalizedSession) -> SessionAnalysis:
    if (
        old.source_id != new.source_id
        or old.corpus != new.corpus
        or old.session_id != new.session_id
        or old.scored_start_sample != new.scored_start_sample
        or old.scored_end_sample != new.scored_end_sample
        or old.source_waveform_sha256 != new.source_waveform_sha256
        or old.annotation_sha256 != new.source_annotation_sha256
    ):
        raise ReferenceMigrationError("v1 and v2 session identities do not match")
    speech, starts, ends, absolute = _speech_correspondence(old.intervals, new.intervals)
    topology = _topology_comparison(old.labels, new.labels)
    row = {
        "schema_version": 1,
        "artifact_role": "reference_migration_session",
        "source_id": old.source_id,
        "corpus": old.corpus,
        "session_id": old.session_id,
        "source_waveform_sha256": old.source_waveform_sha256,
        "source_annotation_sha256": old.annotation_sha256,
        "reference_ref": new.manifest_row()["reference_ref"],
        "reference_sha256": new.reference_sha256,
        "v1_contract_version": old.labels.contract_version,
        "v1_contract_document_sha256": old.labels.contract_document_sha256,
        "v2_contract_version": new.labels.contract_version,
        "v2_contract_document_sha256": new.labels.contract_document_sha256,
        "v1_exposure": _exposure(old.intervals, old.labels),
        "v2_exposure": _exposure(new.intervals, new.labels),
        "speech_correspondence": speech,
        "topology": topology,
        "handoff": _handoff_comparison(old.labels, new.labels),
        "masked_transitions": _mask_comparison(old.labels, new.labels),
        "alignment_risk_counts": {
            "v1_source_clipped_speech_spans": old.clipped_span_count,
            "clipped_tail_rttm_rows": new.parsed_rttm.clipped_tail_row_count,
            "unpaired_v1_speech_segments": speech["unpaired_old_segment_count"],
            "ambiguous_v1_speech_correspondences": speech["ambiguous_old_segment_count"],
            "unpaired_v2_speech_segments": speech["unpaired_new_segment_count"],
        },
        "nonlexical_masks": {
            "count": len(new.parsed_nonlexical.masks),
            "samples": new.labels.exposure["ambiguous_nonlexical_vocalization_samples"],
            "class_counts": dict(new.parsed_nonlexical.observed_class_counts),
            "marker_counts": dict(new.parsed_nonlexical.observed_marker_counts),
        },
        "change_classification": {
            "unchanged_episode_count": topology["unchanged_episode_count"],
            "timing_only_episode_count": topology["timing_only_change_count"],
            "topology_changing_episode_count": topology["topology_changing_total_count"],
        },
    }
    return SessionAnalysis(row, starts, ends, absolute)


def _sum_nested_counts(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> Counter[str]:
    result: Counter[str] = Counter()
    for row in rows:
        value: Any = row
        for field in path:
            value = value[field]
        result.update(value)
    return result


def _aggregate(rows: Sequence[SessionAnalysis]) -> dict[str, Any]:
    raw_rows = [analysis.row for analysis in rows]
    starts = tuple(value for analysis in rows for value in analysis.start_displacements)
    ends = tuple(value for analysis in rows for value in analysis.end_displacements)
    absolute = tuple(value for analysis in rows for value in analysis.absolute_displacements)
    result: dict[str, Any] = {"session_count": len(rows)}
    for version in ("v1", "v2"):
        exposure = {
            field: sum(row[f"{version}_exposure"][field] for row in raw_rows)
            for field in (
                "scored_samples",
                "speech_samples",
                "silence_samples",
                "overlap_samples",
                "reliable_solo_samples",
            )
        }
        result[f"{version}_exposure"] = {
            **exposure,
            **{
                field.replace("_samples", "_hours"): _hours(value)
                for field, value in exposure.items()
            },
        }
    result["speech_correspondence"] = {
        field: sum(row["speech_correspondence"][field] for row in raw_rows)
        for field in (
            "old_speech_segment_count",
            "new_speech_segment_count",
            "deterministic_old_segment_count",
            "ambiguous_old_segment_count",
            "unpaired_old_segment_count",
            "unpaired_new_segment_count",
            "derived_old_speech_samples",
            "removed_internal_pause_samples",
            "removed_outer_padding_samples",
        )
    }
    result["speech_correspondence"].update(
        {
            "removed_internal_pause_hours": _hours(
                result["speech_correspondence"]["removed_internal_pause_samples"]
            ),
            "removed_outer_padding_hours": _hours(
                result["speech_correspondence"]["removed_outer_padding_samples"]
            ),
            "start_displacement_samples": _quantiles(starts),
            "end_displacement_samples": _quantiles(ends),
            "absolute_boundary_displacement_samples": _quantiles(absolute),
        }
    )
    topology_fields = (
        "v1_episode_count",
        "v2_episode_count",
        "matched_identity_within_500ms_count",
        "removed_episode_count",
        "added_episode_count",
        "unchanged_episode_count",
        "timing_only_change_count",
        "topology_changing_match_count",
        "topology_changing_total_count",
    )
    result["topology"] = {
        field: sum(row["topology"][field] for row in raw_rows) for field in topology_fields
    }
    result["topology"]["exact_confusion"] = dict(
        sorted(_sum_nested_counts(raw_rows, ("topology", "exact_confusion")).items())
    )
    aggregate_confusion = _sum_nested_counts(
        raw_rows, ("topology", "direct_gap_overlap_confusion")
    )
    result["topology"]["direct_gap_overlap_confusion"] = {
        f"{old_family}->{new_family}": aggregate_confusion[
            f"{old_family}->{new_family}"
        ]
        for old_family in TOPOLOGY_FAMILIES
        for new_family in TOPOLOGY_FAMILIES
    }
    for version in ("v1", "v2"):
        result["topology"][f"{version}_counts"] = dict(
            sorted(_sum_nested_counts(raw_rows, ("topology", f"{version}_counts")).items())
        )
    result["topology"]["overlap_takeover_return_changes"] = {
        name: result["topology"]["v2_counts"][name] - result["topology"]["v1_counts"][name]
        for name in ("overlap_takeover", "overlap_return")
    }
    result["topology"]["short_backchannel_change"] = (
        result["topology"]["v2_counts"]["short_backchannel_return"]
        - result["topology"]["v1_counts"]["short_backchannel_return"]
    )
    result["topology"]["retention"] = {
        f"within_{threshold_ms}ms": {
            "count": sum(
                row["topology"]["retention"][f"within_{threshold_ms}ms"]["count"]
                for row in raw_rows
            )
        }
        for threshold_ms in MATCH_THRESHOLDS_MS
    }
    for value in result["topology"]["retention"].values():
        value["proportion_of_v1"] = _fraction(
            value["count"], result["topology"]["v1_episode_count"]
        )
    result["handoff"] = {
        field: sum(row["handoff"][field] for row in raw_rows)
        for field in (
            "v1_handoff_count",
            "v2_handoff_count",
            "matched_identity_within_500ms_count",
            "removed_handoff_count",
            "added_handoff_count",
        )
    }
    result["masked_transitions"] = {
        field: sum(row["masked_transitions"][field] for row in raw_rows)
        for field in ("v1_masked_transition_count", "v2_masked_transition_count", "count_change")
    }
    for version in ("v1", "v2"):
        result["masked_transitions"][f"{version}_reasons"] = dict(
            sorted(
                _sum_nested_counts(raw_rows, ("masked_transitions", f"{version}_reasons")).items()
            )
        )
    result["masked_transitions"]["reason_count_changes"] = {
        reason: result["masked_transitions"]["v2_reasons"].get(reason, 0)
        - result["masked_transitions"]["v1_reasons"].get(reason, 0)
        for reason in sorted(
            set(result["masked_transitions"]["v1_reasons"])
            | set(result["masked_transitions"]["v2_reasons"])
        )
    }
    result["alignment_risk_counts"] = {
        field: sum(row["alignment_risk_counts"][field] for row in raw_rows)
        for field in (
            "v1_source_clipped_speech_spans",
            "clipped_tail_rttm_rows",
            "unpaired_v1_speech_segments",
            "ambiguous_v1_speech_correspondences",
            "unpaired_v2_speech_segments",
        )
    }
    result["nonlexical_masks"] = {
        "count": sum(row["nonlexical_masks"]["count"] for row in raw_rows),
        "samples": sum(row["nonlexical_masks"]["samples"] for row in raw_rows),
    }
    result["nonlexical_masks"]["hours"] = _hours(result["nonlexical_masks"]["samples"])
    return result


def build_migration_artifacts(
    old_sessions: Sequence[NormalizedSession],
    new_sessions: Sequence[ReferenceNormalizedSession],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_by_source = {session.source_id: session for session in old_sessions}
    new_by_source = {session.source_id: session for session in new_sessions}
    if (
        len(old_by_source) != len(old_sessions)
        or len(new_by_source) != len(new_sessions)
        or set(old_by_source) != set(new_by_source)
    ):
        raise ReferenceMigrationError("v1 and v2 inventories differ")
    analyses = [
        analyze_session(old_by_source[source_id], new_by_source[source_id])
        for source_id in sorted(old_by_source)
    ]
    rows = [analysis.row for analysis in analyses]
    by_corpus = {
        corpus: _aggregate([analysis for analysis in analyses if analysis.row["corpus"] == corpus])
        for corpus in sorted({analysis.row["corpus"] for analysis in analyses})
    }
    summary = {
        "schema_version": 1,
        "artifact_role": "reference_migration_summary",
        "diagnostic_not_quality_gate": True,
        "model_predictions_consulted": False,
        "matching_policy": {
            "speech_boundary_correspondence": "same-speaker overlap with one old span to one-or-more new spans and no new span shared by multiple old spans",
            "topology_correspondence": "speaker-identity-equal monotonic maximum-cardinality minimum-displacement matching within 500 ms",
            "handoff_correspondence": "speaker-pair-equal monotonic maximum-cardinality minimum-displacement matching within 500 ms",
            "retention_thresholds_ms": list(MATCH_THRESHOLDS_MS),
            "timing_only_definition": "matched identity and unchanged exact topology with changed event time",
            "topology_change_definition": "matched exact-topology change plus unmatched v1 removals and unmatched v2 additions",
        },
        "session_manifest_sha256": canonical_sha256(rows),
        "overall": _aggregate(analyses),
        "by_corpus": by_corpus,
    }
    return rows, summary


def _provenance(
    data_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    new_sessions: Sequence[ReferenceNormalizedSession],
    source_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not new_sessions:
        raise ReferenceMigrationError("reference inventory is empty")
    sources = {row.get("source_id"): row for row in source_rows}
    if len(sources) != len(source_rows) or set(sources) != {
        session.source_id for session in new_sessions
    }:
        raise ReferenceMigrationError("reference and source inventories differ")
    checkout_provenance = dict(new_sessions[0].reference_checkout_provenance)
    if any(
        dict(session.reference_checkout_provenance) != checkout_provenance
        for session in new_sessions
    ):
        raise ReferenceMigrationError("reference sessions do not share one checkout identity")
    references = []
    for session in new_sessions:
        row = session.manifest_row()
        source = sources[session.source_id]
        references.append(
            {
                "source_id": row["source_id"],
                "corpus": row["corpus"],
                "session_id": row["session_id"],
                "reference_ref": row["reference_ref"],
                "reference_sha256": row["reference_sha256"],
                "source_record_sha256": row["source_record_sha256"],
                "source_waveform_sha256": row["source_waveform_sha256"],
                "source_annotation_sha256": row["source_annotation_sha256"],
                "reference_metadata_sha256": row["reference_metadata_sha256"],
                "speaker_mapping_sha256": row["speaker_mapping_sha256"],
                "canonical_intervals_sha256": row["canonical_intervals_sha256"],
                "label_result_sha256": row["label_result_sha256"],
                "source_license_id": source["license_id"],
            }
        )
    references.sort(key=lambda row: row["source_id"])
    contract_v0 = load_contract(version="psem-handoff-v0")
    contract_v1 = load_contract(version="psem-handoff-v1")
    input_paths = {
        "operational_label_contract.json": CONTRACT_PATH_BY_VERSION["psem-handoff-v0"],
        "v2/annotation_manifest.jsonl": data_dir / "v2" / "annotation_manifest.jsonl",
        "v2/nonlexical_risk_inventory.json": INVENTORY_PATH,
        "v2/normalization_manifest.jsonl": data_dir / "v2" / "normalization_manifest.jsonl",
        "v2/operational_label_contract.json": CONTRACT_PATH_BY_VERSION["psem-handoff-v1"],
        "v2/prior_exposure_manifest.jsonl": data_dir / "v2" / "prior_exposure_manifest.jsonl",
        "v2/source_manifest.jsonl": data_dir / "v2" / "source_manifest.jsonl",
    }
    input_artifacts = []
    for ref, consumed_path in sorted(input_paths.items()):
        declared_path = data_dir / ref
        if sha256_file(declared_path) != sha256_file(consumed_path):
            raise ReferenceMigrationError(
                f"declared input differs from installed pipeline input: {ref}"
            )
        input_artifacts.append({"ref": ref, "sha256": sha256_file(consumed_path)})
    source_license_ids_by_corpus = {
        corpus: sorted(
            {
                source["license_id"]
                for source in source_rows
                if source.get("corpus") == corpus
            }
        )
        for corpus in ("AMI", "AliMeeting")
    }
    return {
        "schema_version": 1,
        "artifact_role": "reference_provenance",
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "reference_git_tree": checkout_provenance["git_tree"],
        "reference_license_ref": checkout_provenance["license_ref"],
        "reference_license_sha256": checkout_provenance["license_sha256"],
        "source_license_ids_by_corpus": source_license_ids_by_corpus,
        "v1_contract_document_sha256": contract_v0.document_sha256,
        "v2_contract_document_sha256": contract_v1.document_sha256,
        "nonlexical_inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "input_artifacts": input_artifacts,
        "source_count": len(references),
        "reference_inventory_sha256": canonical_sha256(references),
        "migration_session_manifest_sha256": canonical_sha256(list(rows)),
        "references": references,
    }


def _integrity_report(
    data_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    old_sessions: Sequence[NormalizedSession],
    new_sessions: Sequence[ReferenceNormalizedSession],
    source_rows: Sequence[Mapping[str, Any]],
    checked_in_rows: Sequence[Mapping[str, Any]],
    regenerated_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    source_by_id = {row.get("source_id"): row for row in source_rows}
    old_by_id = {session.source_id: session for session in old_sessions}
    new_by_id = {session.source_id: session for session in new_sessions}
    migration_by_id = {row.get("source_id"): row for row in rows}
    reference_by_id = {
        row.get("source_id"): row for row in provenance["references"]
    }
    selection = validate_selection_receipt(data_dir / "alimeeting_train_selection.json")
    selected_train_ids = {
        f"alimeeting_{session_id}" for session_id in selection["selected_session_ids"]
    }
    expected_source_ids = (
        {f"ami_{session_id}" for session_id in EXPECTED_AMI_MEETINGS}
        | {f"alimeeting_{session_id}" for session_id in EXPECTED_ALIMEETING_MEETINGS}
        | selected_train_ids
    )
    reference_hashes = [row["reference_sha256"] for row in provenance["references"]]
    reference_refs = [row["reference_ref"] for row in provenance["references"]]
    clipped = sum(session.parsed_rttm.clipped_tail_row_count for session in new_sessions)
    train_clipped = sum(
        session.parsed_rttm.clipped_tail_row_count
        for session in new_sessions
        if session.corpus == "AliMeeting"
        and "Train_Ali_far" in session.manifest_row()["reference_ref"]
    )
    exact_inventory = all(
        len(collection) == len(expected_source_ids)
        and set(collection) == expected_source_ids
        for collection in (
            source_by_id,
            old_by_id,
            new_by_id,
            migration_by_id,
            reference_by_id,
        )
    )
    checkout = dict(new_sessions[0].reference_checkout_provenance) if new_sessions else {}
    exact_upstream = bool(new_sessions) and all(
        dict(session.reference_checkout_provenance) == checkout for session in new_sessions
    ) and (
        checkout.get("repository") == REFERENCE_REPOSITORY
        and checkout.get("commit") == REFERENCE_COMMIT
        and provenance.get("reference_repository") == checkout.get("repository")
        and provenance.get("reference_commit") == checkout.get("commit")
        and provenance.get("reference_git_tree") == checkout.get("git_tree")
        and provenance.get("reference_license_ref") == checkout.get("license_ref")
        and provenance.get("reference_license_sha256") == checkout.get("license_sha256")
    )
    references_bound = exact_inventory and all(
        reference_by_id[source_id]["reference_ref"]
        == new_by_id[source_id].manifest_row()["reference_ref"]
        == migration_by_id[source_id]["reference_ref"]
        and reference_by_id[source_id]["reference_ref"]
        == source_by_id[source_id].get(
            "reference_ref", reference_by_id[source_id]["reference_ref"]
        )
        and reference_by_id[source_id]["reference_sha256"]
        == new_by_id[source_id].reference_sha256
        == migration_by_id[source_id]["reference_sha256"]
        and reference_by_id[source_id]["reference_sha256"]
        == source_by_id[source_id].get(
            "reference_sha256", reference_by_id[source_id]["reference_sha256"]
        )
        and reference_by_id[source_id]["source_record_sha256"]
        == canonical_sha256(dict(source_by_id[source_id]))
        and reference_by_id[source_id]["source_license_id"]
        == source_by_id[source_id]["license_id"]
        for source_id in expected_source_ids
    )
    source_licenses_bound = exact_inventory and all(
        source_by_id[source_id]["license_id"]
        == ("CC-BY-4.0" if source_by_id[source_id]["corpus"] == "AMI" else "CC-BY-SA-4.0")
        for source_id in expected_source_ids
    ) and provenance.get("source_license_ids_by_corpus") == {
        "AMI": ["CC-BY-4.0"],
        "AliMeeting": ["CC-BY-SA-4.0"],
    }
    speaker_mapping_valid = exact_inventory and all(
        re.fullmatch(r"[0-9a-f]{64}", session.speaker_mapping_sha256)
        and all(interval.speaker_identity_known for interval in session.intervals)
        and {
            speaker_id
            for span in session.parsed_rttm.spans
            for speaker_id in (span.speaker_id,)
        }.issubset(set(source_by_id[session.source_id]["speaker_ids"]))
        and {
            speaker_id
            for interval in session.intervals
            for speaker_id in interval.active_speakers
        }.issubset(set(source_by_id[session.source_id]["speaker_ids"]))
        for session in new_sessions
    ) and all(
        all(interval.speaker_identity_known for interval in old_by_id[source_id].intervals)
        and {
            speaker_id
            for interval in old_by_id[source_id].intervals
            for speaker_id in interval.active_speakers
        }.issubset(set(source_by_id[source_id]["speaker_ids"]))
        and migration_by_id[source_id]["v1_exposure"]["reliable_solo_samples"] > 0
        and migration_by_id[source_id]["topology"]["v1_episode_count"] > 0
        for source_id in selected_train_ids
    )
    rttm_timing_valid = all(
        session.parsed_rttm.raw_row_count >= len(session.parsed_rttm.spans) > 0
        and all(
            session.scored_start_sample <= span.start_sample < span.end_sample
            <= session.scored_end_sample
            for span in session.parsed_rttm.spans
        )
        for session in new_sessions
    )
    nonlexical_valid = all(
        session.inventory_sha256 == EXPECTED_INVENTORY_SHA256
        and all(
            mask.mask_class == MASK_CLASS
            and session.scored_start_sample <= mask.start_sample < mask.end_sample
            <= session.scored_end_sample
            and mask.speaker_id in set(source_by_id[session.source_id]["speaker_ids"])
            and bool(mask.source_annotation_id)
            and bool(mask.annotation_class)
            for mask in session.parsed_nonlexical.masks
        )
        and all(count >= 0 for count in session.parsed_nonlexical.observed_class_counts.values())
        and all(count >= 0 for count in session.parsed_nonlexical.observed_marker_counts.values())
        for session in new_sessions
    )
    canonical_timeline_valid = all(
        bool(session.intervals)
        and session.intervals[0].start_sample == session.scored_start_sample
        and session.intervals[-1].end_sample == session.scored_end_sample
        and all(
            left.end_sample == right.start_sample
            for left, right in zip(session.intervals, session.intervals[1:])
        )
        and tuple(session.labels.intervals) == session.intervals
        for session in new_sessions
    )
    v1_contract = load_contract(version="psem-handoff-v1")
    topology_deterministic = all(
        generate_labels(
            session.intervals,
            contract=v1_contract,
            scored_start_sample=session.scored_start_sample,
            scored_end_sample=session.scored_end_sample,
        ).to_dict()
        == session.labels.to_dict()
        for session in new_sessions
    )
    repeat_rows, repeat_summary = build_migration_artifacts(old_sessions, new_sessions)
    deterministic_rebuild = list(rows) == repeat_rows and dict(summary) == repeat_summary
    expected_tail_receipts = {
        "alimeeting_R2001_M2205": 159,
        "alimeeting_R2001_M2206": 159,
    }
    tail_receipts_valid = exact_inventory and all(
        source_by_id[source_id].get("annotation_tail_excess_samples", 0)
        == expected_tail_receipts.get(source_id, 0)
        for source_id in expected_source_ids
    )
    clipped_sessions = {
        session.source_id: session.parsed_rttm.clipped_tail_row_count
        for session in new_sessions
        if session.parsed_rttm.clipped_tail_row_count
    }
    checks = {
        "exact_upstream_revision": exact_upstream,
        "complete_expected_unique_source_inventory": exact_inventory,
        "every_reference_hash_present_and_bound": references_bound
        and len(reference_hashes) == len(expected_source_ids)
        and all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in reference_hashes
        ),
        "every_reference_identity_unique": len(reference_refs) == len(expected_source_ids)
        and len(set(reference_refs)) == len(expected_source_ids),
        "source_licenses_present_and_bound": source_licenses_bound,
        "speaker_mapping_fail_closed": speaker_mapping_valid,
        "rttm_timing_validated": rttm_timing_valid,
        "unknown_nonlexical_class_fail_closed": nonlexical_valid,
        "canonical_timeline_deterministic": canonical_timeline_valid
        and list(checked_in_rows) == list(regenerated_rows),
        "topology_generation_deterministic": topology_deterministic,
        "migration_rebuild_deterministic": deterministic_rebuild,
        "checked_in_v2_normalization_matches_regeneration": list(checked_in_rows)
        == list(regenerated_rows),
        "declared_source_tail_receipts_exact": tail_receipts_valid,
        "selected_train_rttm_clipping_zero": train_clipped == 0,
        "only_accepted_ami_terminal_tail_clips": clipped_sessions
        == {"ami_ES2010b": 1, "ami_ES2013c": 1},
    }
    return {
        "schema_version": 1,
        "artifact_role": "reference_integrity_report",
        "scope": "pipeline_correctness_not_independent_acoustic_boundary_accuracy",
        "input_policy": {
            "model_predictions_or_scores_accepted": False,
            "selection_receipt_model_inputs": selection["selection_model_inputs"],
        },
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "source_count": len(rows),
        "reference_count": len(reference_hashes),
        "raw_rttm_row_count": sum(session.parsed_rttm.raw_row_count for session in new_sessions),
        "canonical_rttm_span_count": sum(
            len(session.parsed_rttm.spans) for session in new_sessions
        ),
        "clipped_tail_rttm_row_count": clipped,
        "selected_train_clipped_tail_rttm_row_count": train_clipped,
        "nonlexical_mask_count": sum(
            len(session.parsed_nonlexical.masks) for session in new_sessions
        ),
        "reference_inventory_sha256": provenance["reference_inventory_sha256"],
        "migration_session_manifest_sha256": provenance["migration_session_manifest_sha256"],
        "reference_provenance_sha256": canonical_sha256(dict(provenance)),
        "migration_summary_sha256": canonical_sha256(dict(summary)),
    }


def _render_diagnostic_detail(title: str, row: Mapping[str, Any], level: int) -> list[str]:
    speech = row["speech_correspondence"]
    topology = row["topology"]
    handoff = row["handoff"]
    masks = row["masked_transitions"]
    risks = row["alignment_risk_counts"]
    nonlexical = row["nonlexical_masks"]
    lines = [f"{'#' * level} {title}", ""]
    lines.extend(
        [
            "| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for version in ("v1", "v2"):
        exposure = row[f"{version}_exposure"]
        lines.append(
            f"| {version} | {exposure['speech_hours']:.6f} | {exposure['silence_hours']:.6f} | {exposure['overlap_hours']:.6f} | {exposure['reliable_solo_hours']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Speech segments v1/v2: {speech['old_speech_segment_count']} / {speech['new_speech_segment_count']}",
            f"- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: {speech['deterministic_old_segment_count']} / {speech['ambiguous_old_segment_count']} / {speech['unpaired_old_segment_count']} / {speech['unpaired_new_segment_count']}",
            f"- Removed internal pause: {speech['removed_internal_pause_samples']} samples ({_hours(speech['removed_internal_pause_samples']):.6f} h)",
            f"- Removed outer padding: {speech['removed_outer_padding_samples']} samples ({_hours(speech['removed_outer_padding_samples']):.6f} h)",
            "",
            "| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, key in (
        ("Start samples", "start_displacement_samples"),
        ("End samples", "end_displacement_samples"),
        ("Absolute samples", "absolute_boundary_displacement_samples"),
    ):
        quantiles = speech[key]
        lines.append(
            f"| {label} | {quantiles['p00']} | {quantiles['p25']} | {quantiles['p50']} | {quantiles['p75']} | {quantiles['p90']} | {quantiles['p95']} | {quantiles['p99']} | {quantiles['p100']} |"
        )
    lines.extend(
        [
            "",
            "| v1 → v2 family | Direct | Gap | Overlap |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    confusion = topology["direct_gap_overlap_confusion"]
    for old_family in TOPOLOGY_FAMILIES:
        lines.append(
            f"| {old_family.title()} | {confusion[f'{old_family}->direct']} | {confusion[f'{old_family}->gap']} | {confusion[f'{old_family}->overlap']} |"
        )
    lines.extend(
        [
            "",
            f"- Topology episodes v1/v2/matched/added/removed: {topology['v1_episode_count']} / {topology['v2_episode_count']} / {topology['matched_identity_within_500ms_count']} / {topology['added_episode_count']} / {topology['removed_episode_count']}",
            f"- Unchanged/timing-only/topology-changing: {topology['unchanged_episode_count']} / {topology['timing_only_change_count']} / {topology['topology_changing_total_count']}",
            f"- Overlap takeover/return changes: {topology['overlap_takeover_return_changes']['overlap_takeover']} / {topology['overlap_takeover_return_changes']['overlap_return']}",
            f"- Short-backchannel change: {topology['short_backchannel_change']}",
            f"- Handoffs v1/v2/matched/added/removed: {handoff['v1_handoff_count']} / {handoff['v2_handoff_count']} / {handoff['matched_identity_within_500ms_count']} / {handoff['added_handoff_count']} / {handoff['removed_handoff_count']}",
            "",
            "| Retention collar | Count | Proportion of v1 |",
            "| --- | ---: | ---: |",
        ]
    )
    for threshold_ms in MATCH_THRESHOLDS_MS:
        retention = topology["retention"][f"within_{threshold_ms}ms"]
        lines.append(
            f"| {threshold_ms} ms | {retention['count']} | {retention['proportion_of_v1']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Masked transitions v1/v2/change: {masks['v1_masked_transition_count']} / {masks['v2_masked_transition_count']} / {masks['count_change']}",
            f"- Mask reasons v1: `{json.dumps(masks['v1_reasons'], ensure_ascii=False, sort_keys=True, separators=(',', ':'))}`",
            f"- Mask reasons v2: `{json.dumps(masks['v2_reasons'], ensure_ascii=False, sort_keys=True, separators=(',', ':'))}`",
            f"- Mask reason changes: `{json.dumps(masks['reason_count_changes'], ensure_ascii=False, sort_keys=True, separators=(',', ':'))}`",
            f"- Alignment risks: `{json.dumps(risks, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}`",
            f"- Nonlexical masks: {nonlexical['count']} / {nonlexical['samples']} samples / {_hours(nonlexical['samples']):.6f} h",
            "",
        ]
    )
    return lines


def render_reference_migration(
    summary: Mapping[str, Any],
    integrity: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    overall = summary["overall"]
    lines = [
        "# PSEM v1 to v2 reference migration",
        "",
        "This is a deterministic diagnostic migration report, not a boundary-quality gate. No model predictions or scores participate.",
        "",
        "PSEM-STRATEGY-DATA-v2 adopts the commit-pinned forced-aligned AMI/AliMeeting references released by Horiguchi et al. (ASRU 2025) as the common temporal activity reference. This project does not perform additional manual boundary adjudication or independently estimate their acoustic boundary accuracy.",
        "",
        "## Scope",
        "",
        f"- Sessions: {overall['session_count']}",
        f"- AMI: {summary['by_corpus']['AMI']['session_count']}",
        f"- AliMeeting: {summary['by_corpus']['AliMeeting']['session_count']}",
        f"- Reference integrity: `{integrity['status']}`",
        "",
        "## Exposure",
        "",
        "| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for version in ("v1", "v2"):
        exposure = overall[f"{version}_exposure"]
        lines.append(
            f"| {version} | {exposure['speech_hours']:.6f} | {exposure['silence_hours']:.6f} | {exposure['overlap_hours']:.6f} | {exposure['reliable_solo_hours']:.6f} |"
        )
    lines.extend(
        [
            "",
            "### Exposure by corpus",
            "",
            "| Corpus | Reference | Speech h | Silence h | Overlap h | Reliable-solo h |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for corpus in sorted(summary["by_corpus"]):
        for version in ("v1", "v2"):
            exposure = summary["by_corpus"][corpus][f"{version}_exposure"]
            lines.append(
                f"| {corpus} | {version} | {exposure['speech_hours']:.6f} | {exposure['silence_hours']:.6f} | {exposure['overlap_hours']:.6f} | {exposure['reliable_solo_hours']:.6f} |"
            )
    speech = overall["speech_correspondence"]
    lines.extend(
        [
            "",
            "## Boundary and source-span migration",
            "",
            f"- Deterministic old-span correspondences: {speech['deterministic_old_segment_count']}",
            f"- Derived internal-pause removal: {speech['removed_internal_pause_hours']:.6f} h",
            f"- Derived outer-padding removal: {speech['removed_outer_padding_hours']:.6f} h",
            f"- Absolute boundary displacement p50/p90/p99 samples: {speech['absolute_boundary_displacement_samples']['p50']} / {speech['absolute_boundary_displacement_samples']['p90']} / {speech['absolute_boundary_displacement_samples']['p99']}",
            "",
            "## Topology migration",
            "",
        ]
    )
    topology = overall["topology"]
    lines.extend(
        [
            f"- v1/v2 exclusive episodes: {topology['v1_episode_count']} / {topology['v2_episode_count']}",
            f"- Unchanged episodes: {topology['unchanged_episode_count']}",
            f"- Timing-only changes: {topology['timing_only_change_count']}",
            f"- Topology-changing matches/additions/removals: {topology['topology_changing_total_count']}",
            f"- Handoff additions/removals: {overall['handoff']['added_handoff_count']} / {overall['handoff']['removed_handoff_count']}",
            f"- Short-backchannel net change: {topology['short_backchannel_change']}",
            f"- Overlap takeover/return net changes: {topology['overlap_takeover_return_changes']['overlap_takeover']} / {topology['overlap_takeover_return_changes']['overlap_return']}",
            "",
            "| Retention collar | Retained v1 identity and event time | Proportion |",
            "| --- | ---: | ---: |",
        ]
    )
    for threshold_ms in MATCH_THRESHOLDS_MS:
        retention = topology["retention"][f"within_{threshold_ms}ms"]
        lines.append(
            f"| {threshold_ms} ms | {retention['count']} | {retention['proportion_of_v1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "### Direct/gap/overlap confusion",
            "",
            "| v1 → v2 topology family | Matched episodes |",
            "| --- | ---: |",
        ]
    )
    for transition, count in topology["direct_gap_overlap_confusion"].items():
        lines.append(f"| {transition.replace('->', ' → ')} | {count} |")
    lines.extend(
        [
            "",
            "### Topology by corpus",
            "",
            "| Corpus | v1/v2 episodes | Timing-only | Topology-changing | Retained ≤50/500 ms | Handoff +/− | Masked v1/v2 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for corpus in sorted(summary["by_corpus"]):
        corpus_row = summary["by_corpus"][corpus]
        corpus_topology = corpus_row["topology"]
        lines.append(
            f"| {corpus} | {corpus_topology['v1_episode_count']}/{corpus_topology['v2_episode_count']} | {corpus_topology['timing_only_change_count']} | {corpus_topology['topology_changing_total_count']} | {corpus_topology['retention']['within_50ms']['proportion_of_v1']:.6f}/{corpus_topology['retention']['within_500ms']['proportion_of_v1']:.6f} | {corpus_row['handoff']['added_handoff_count']}/{corpus_row['handoff']['removed_handoff_count']} | {corpus_row['masked_transitions']['v1_masked_transition_count']}/{corpus_row['masked_transitions']['v2_masked_transition_count']} |"
        )
    lines.extend(
        [
            "",
            "## Masks and integrity",
            "",
            f"- v1/v2 masked transitions: {overall['masked_transitions']['v1_masked_transition_count']} / {overall['masked_transitions']['v2_masked_transition_count']}",
            f"- v2 nonlexical masks: {overall['nonlexical_masks']['count']} ({overall['nonlexical_masks']['hours']:.6f} h)",
            f"- RTTM terminal-tail clips: {integrity['clipped_tail_rttm_row_count']} total, {integrity['selected_train_clipped_tail_rttm_row_count']} selected Train",
            "",
            "## Per-meeting comparison",
            "",
            "| Source | v1/v2 speech h | v1/v2 overlap h | v1/v2 reliable-solo h | Retained ≤50/500 ms | Timing-only/topology-changing | Handoff +/− | Nonlexical masks |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            "",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['source_id']}` | {row['v1_exposure']['speech_hours']:.6f}/{row['v2_exposure']['speech_hours']:.6f} | {row['v1_exposure']['overlap_hours']:.6f}/{row['v2_exposure']['overlap_hours']:.6f} | {row['v1_exposure']['reliable_solo_hours']:.6f}/{row['v2_exposure']['reliable_solo_hours']:.6f} | {row['topology']['retention']['within_50ms']['proportion_of_v1']:.6f}/{row['topology']['retention']['within_500ms']['proportion_of_v1']:.6f} | {row['topology']['timing_only_change_count']}/{row['topology']['topology_changing_total_count']} | {row['handoff']['added_handoff_count']}/{row['handoff']['removed_handoff_count']} | {row['nonlexical_masks']['count']} |"
        )
    lines.extend(
        [
            "",
            "## Corpus diagnostic detail",
            "",
        ]
    )
    for corpus in sorted(summary["by_corpus"]):
        lines.extend(_render_diagnostic_detail(corpus, summary["by_corpus"][corpus], 3))
    lines.extend(["## Meeting diagnostic detail", ""])
    for row in rows:
        lines.extend(_render_diagnostic_detail(f"`{row['source_id']}`", row, 3))
    return "\n".join(lines)


def write_reference_migration(
    data_dir: Path,
    corpus_root: Path,
    reference_root: Path,
    output_dir: Path,
) -> None:
    old_sessions = normalize_inventory(data_dir / "v2", corpus_root)
    source_rows = _load_jsonl(data_dir / "v2" / "source_manifest.jsonl")
    new_sessions = normalize_reference_inventory(
        data_dir / "v2" / "source_manifest.jsonl", corpus_root, reference_root
    )
    checked_in_rows = _load_jsonl(data_dir / "v2" / "normalization_manifest.jsonl")
    regenerated_rows = [session.manifest_row() for session in new_sessions]
    if checked_in_rows != regenerated_rows:
        raise ReferenceMigrationError("checked-in v2 normalization differs from regeneration")
    rows, summary = build_migration_artifacts(old_sessions, new_sessions)
    provenance = _provenance(data_dir, rows, new_sessions, source_rows)
    integrity = _integrity_report(
        data_dir,
        rows,
        old_sessions,
        new_sessions,
        source_rows,
        checked_in_rows,
        regenerated_rows,
        summary,
        provenance,
    )
    if integrity["status"] != "pass":
        failed = [name for name, passed in integrity["checks"].items() if not passed]
        raise ReferenceMigrationError(f"reference integrity failed: {', '.join(failed)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "reference_migration.jsonl", rows)
    _write_json(output_dir / "reference_migration_summary.json", summary)
    _write_json(output_dir / "reference_provenance.json", provenance)
    _write_json(output_dir / "reference_integrity_report.json", integrity)
    (output_dir / "REFERENCE_MIGRATION.md").write_text(
        render_reference_migration(summary, integrity, rows),
        encoding="utf-8",
        newline="\n",
    )
    artifact_names = (
        "reference_migration.jsonl",
        "reference_migration_summary.json",
        "reference_provenance.json",
        "reference_integrity_report.json",
        "REFERENCE_MIGRATION.md",
    )
    artifact_sha256 = {
        name: sha256_file(output_dir / name) for name in artifact_names
    }
    _write_json(
        output_dir / "reference_artifact_receipt.json",
        {
            "schema_version": 1,
            "artifact_role": "reference_migration_artifact_receipt",
            "source_count": len(rows),
            "artifact_sha256": artifact_sha256,
            "artifact_set_sha256": canonical_sha256(artifact_sha256),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_reference_migration(
        args.data_dir.resolve(),
        args.corpus_root.resolve(),
        args.reference_root.resolve(),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
