from __future__ import annotations

import argparse
import tempfile
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from experiments.psem_relative_occupancy_gate.contracts import evaluation_cells
from experiments.psem_relative_occupancy_gate.io_utils import (
    CONFIG_PATH,
    canonical_sha256,
    config,
    load_json,
    load_jsonl,
    safe_output_path,
    sha256_file,
    write_json,
)
from experiments.psem_relative_occupancy_gate.model_evaluate import (
    gt_reference_session,
    gt_singleton_opportunities,
    intervals_from_manifest,
)
from experiments.psem_relative_occupancy_gate.provenance import load_frozen_dataset
from experiments.psem_relative_occupancy_gate.run_gate1 import run as run_gate1
from experiments.psem_relative_occupancy_gate.run_gate2 import run as run_gate2


class ModelGateVerificationError(RuntimeError):
    pass


def _load_object(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ModelGateVerificationError(f"verification artifact is not an object: {path}")
    return value


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    p90 = row["metrics"]["enrollment_delay_ms"]["p90"]
    return (
        float(row["metrics"]["wrong_anchor_rate"]),
        -float(row["metrics"]["fraction_enrolled_within_1500ms"]),
        float(row["metrics"]["enrollment_failure_rate"]),
        float(p90) if p90 is not None else float("inf"),
        -float(row["config"]["active_threshold"]),
        float(row["config"]["other_low_threshold"]),
        float(row["config"]["confirmation_samples"]),
    )


def _validate_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    expected_ids = sorted(load_frozen_dataset().source_ids("PSEM-STRATEGY-DEV"))
    observed_ids = sorted(str(row.get("source_id", "")) for row in rows)
    if observed_ids != expected_ids or len(observed_ids) != len(set(observed_ids)):
        raise ModelGateVerificationError("DEV manifest source coverage mismatch")
    for row in rows:
        payload = dict(row)
        observed_hash = payload.pop("row_sha256", None)
        if (
            observed_hash != canonical_sha256(payload)
            or row.get("schema_version") != "psem.relative_occupancy.manifest_row.v1"
            or row.get("role") != "PSEM-STRATEGY-DEV"
            or row.get("eval_status") != "sealed"
            or row.get("eval_selection_sha256") is not None
            or row.get("eval_authorization_sha256") is not None
            or row.get("config_sha256") != sha256_file(CONFIG_PATH)
        ):
            raise ModelGateVerificationError(
                f"DEV manifest binding mismatch: {row.get('source_id')}"
            )
    return {str(row["source_id"]): row for row in rows}


def _validate_row_hash(row: dict[str, Any]) -> None:
    payload = dict(row)
    observed_hash = payload.pop("row_sha256", None)
    if observed_hash != canonical_sha256(payload):
        raise ModelGateVerificationError("event-ledger row hash mismatch")


def _validate_replacement_event(
    event: dict[str, Any], *, source_id: str, source_end: int, confirmation_samples: int
) -> None:
    if event.get("source_id") != source_id:
        raise ModelGateVerificationError("event source binding mismatch")
    try:
        boundary = int(event["boundary_source_sample"])
        frontier = int(event["model_evidence_frontier_sample"])
        emit = int(event["decoder_emit_sample"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelGateVerificationError("event timing fields are invalid") from exc
    if (
        not 0 <= boundary <= frontier <= emit <= source_end
        or event.get("confirmation_samples") != confirmation_samples
    ):
        raise ModelGateVerificationError("event timing contract mismatch")


def _validate_exposure(value: dict[str, Any]) -> None:
    fields = (
        "masked_seconds",
        "masked_active_speech_seconds",
        "unanchored_active_speech_seconds",
        "anchor_uncertain_active_speech_seconds",
        "fail_closed_unknown_active_speech_seconds",
        "exclusive_other_contamination_upper_bound_seconds",
    )
    if any(
        field not in value
        or not isinstance(value[field], (int, float))
        or float(value[field]) < 0.0
        for field in fields
    ):
        raise ModelGateVerificationError("fail-closed exposure contract mismatch")
    expected_unknown = (
        float(value["masked_active_speech_seconds"])
        + float(value["unanchored_active_speech_seconds"])
        + float(value["anchor_uncertain_active_speech_seconds"])
    )
    if (
        abs(float(value["fail_closed_unknown_active_speech_seconds"]) - expected_unknown)
        > 1e-9
    ):
        raise ModelGateVerificationError("fail-closed unknown exposure is inconsistent")


def _assert_exposure_matches(
    observed: dict[str, Any], expected: dict[str, float]
) -> None:
    for field, value in expected.items():
        if field not in observed or abs(float(observed[field]) - value) > 1e-9:
            raise ModelGateVerificationError(
                f"fail-closed exposure replay mismatch: {field}"
            )


def _range_coverage(start: int, end: int, ranges: Sequence[tuple[int, int]]) -> int:
    return sum(max(0, min(end, right) - max(start, left)) for left, right in ranges)


def _gate1_exposure_replay(
    row: dict[str, Any], manifest_row: dict[str, Any]
) -> dict[str, float]:
    cfg = config()
    intervals = intervals_from_manifest(manifest_row)
    reference = gt_reference_session(
        manifest_row,
        replacement_confirmation_samples=int(row["replacement_confirm_ms"]) * 16,
        enrollment_samples=int(cfg["gate0_enrollment_confirm_ms"]) * 16,
        silence_reset_samples=int(cfg["lifecycle_proxy_silence_reset_ms"]) * 16,
    )
    if row.get("reference_events") != [value.to_dict() for value in reference.events]:
        raise ModelGateVerificationError("Gate 1 reference replay mismatch")
    mapping_ids = {
        str(value["anchor_episode_id"]) for value in row.get("oracle_mappings", [])
    }
    events = {
        str(value["anchor_episode_id"]): value for value in row.get("events", [])
    }
    anchored_ranges: list[tuple[int, int]] = []
    uncertain_ranges: list[tuple[int, int]] = []
    reference_ids = {value.episode_id for value in reference.episodes}
    if not mapping_ids <= reference_ids:
        raise ModelGateVerificationError("Gate 1 mapping references an unknown episode")
    for episode in reference.episodes:
        if episode.episode_id not in mapping_ids:
            uncertain_ranges.append((episode.anchor_emit_sample, episode.end_emit_sample))
            continue
        event = events.get(episode.episode_id)
        end = min(
            episode.end_emit_sample,
            int(event["decoder_emit_sample"]) if event is not None else episode.end_emit_sample,
        )
        anchored_ranges.append((episode.anchor_emit_sample, end))
    masked = 0
    masked_active = 0
    unanchored_active = 0
    uncertain_active = 0
    for interval in intervals:
        duration = interval.end_sample - interval.start_sample
        if interval.masked:
            masked += duration
            if interval.active_speakers:
                masked_active += duration
            continue
        if not interval.active_speakers:
            continue
        anchored = _range_coverage(
            interval.start_sample, interval.end_sample, anchored_ranges
        )
        uncertain = _range_coverage(
            interval.start_sample, interval.end_sample, uncertain_ranges
        )
        uncertain_active += uncertain
        unanchored_active += max(0, duration - anchored - uncertain)
    exact = float(row["fail_closed_exposure"]["exclusive_other_contamination_seconds"])
    return {
        "masked_seconds": masked / 16000.0,
        "masked_active_speech_seconds": masked_active / 16000.0,
        "unanchored_active_speech_seconds": unanchored_active / 16000.0,
        "anchor_uncertain_active_speech_seconds": uncertain_active / 16000.0,
        "fail_closed_unknown_active_speech_seconds": (
            masked_active + unanchored_active + uncertain_active
        )
        / 16000.0,
        "exclusive_other_contamination_seconds": exact,
        "exclusive_other_contamination_upper_bound_seconds": (
            exact + unanchored_active / 16000.0 + uncertain_active / 16000.0
        ),
    }


def _gate2_exposure_replay(row: dict[str, Any]) -> dict[str, float]:
    masked = 0
    masked_active = 0
    unanchored_active = 0
    uncertain_active = 0
    for span in row["timeline"]:
        duration = int(span["end_sample"]) - int(span["start_sample"])
        if span.get("masked") is True:
            masked += duration
            if span.get("speech_present") is True:
                masked_active += duration
        elif span.get("state") is None and span.get("speech_present") is True:
            if span.get("lifecycle") == "UNANCHORED":
                unanchored_active += duration
            elif span.get("lifecycle") == "ANCHOR_UNCERTAIN":
                uncertain_active += duration
    exact = float(row["fail_closed_exposure"]["exclusive_other_contamination_seconds"])
    return {
        "masked_seconds": masked / 16000.0,
        "masked_active_speech_seconds": masked_active / 16000.0,
        "unanchored_active_speech_seconds": unanchored_active / 16000.0,
        "anchor_uncertain_active_speech_seconds": uncertain_active / 16000.0,
        "fail_closed_unknown_active_speech_seconds": (
            masked_active + unanchored_active + uncertain_active
        )
        / 16000.0,
        "exclusive_other_contamination_seconds": exact,
        "exclusive_other_contamination_upper_bound_seconds": (
            exact + unanchored_active / 16000.0 + uncertain_active / 16000.0
        ),
    }


def _causal_opportunity_replay(
    row: dict[str, Any], manifest_row: dict[str, Any], cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    intervals = intervals_from_manifest(manifest_row)
    enrollment_samples = int(cfg["gate0_enrollment_confirm_ms"]) * 16
    silence_reset_samples = int(cfg["lifecycle_proxy_silence_reset_ms"]) * 16
    scored_start = int(manifest_row["scored_start_sample"])
    scored_end = int(manifest_row["scored_end_sample"])
    episodes = row["annotated_episodes"]
    result: list[dict[str, Any]] = []
    window_start = scored_start
    for episode in episodes:
        window_end = min(
            max(int(episode["anchor_emit_sample"]), window_start + 1), scored_end
        )
        opportunities = gt_singleton_opportunities(
            intervals,
            start_sample=window_start,
            end_sample=window_end,
            enrollment_samples=enrollment_samples,
            silence_reset_samples=silence_reset_samples,
        )
        for index, (speaker, opportunity_start, opportunity_emit) in enumerate(
            opportunities
        ):
            result.append(
                {
                    "window_start_sample": window_start,
                    "window_end_sample": window_end,
                    "expected_anchor_speaker": speaker,
                    "opportunity_start_sample": opportunity_start,
                    "opportunity_emit_sample": opportunity_emit,
                    "matched_anchor_episode_id": (
                        str(episode["episode_id"])
                        if index == len(opportunities) - 1
                        else None
                    ),
                }
            )
        expected = opportunities[-1] if opportunities else None
        if episode.get("expected_anchor_speaker") != (
            expected[0] if expected is not None else None
        ) or episode.get("opportunity_start_sample") != (
            expected[1] if expected is not None else None
        ):
            raise ModelGateVerificationError(
                "causal episode is not aligned to its latest lifecycle opportunity"
            )
        window_start = min(
            max(int(episode["end_emit_sample"]), window_start), scored_end
        )
    if window_start < scored_end:
        for speaker, opportunity_start, opportunity_emit in gt_singleton_opportunities(
            intervals,
            start_sample=window_start,
            end_sample=scored_end,
            enrollment_samples=enrollment_samples,
            silence_reset_samples=silence_reset_samples,
        ):
            result.append(
                {
                    "window_start_sample": window_start,
                    "window_end_sample": scored_end,
                    "expected_anchor_speaker": speaker,
                    "opportunity_start_sample": opportunity_start,
                    "opportunity_emit_sample": opportunity_emit,
                    "matched_anchor_episode_id": None,
                }
            )
    return result


def _validate_gate1_row(row: dict[str, Any], manifest_row: dict[str, Any]) -> None:
    confirmation_samples = int(row["replacement_confirm_ms"]) * 16
    source_id = str(row["source_id"])
    source_end = int(manifest_row["source_duration_samples"])
    mappings = row.get("oracle_mappings")
    events = row.get("events")
    references = row.get("reference_events")
    exposure = row.get("fail_closed_exposure")
    if not all(isinstance(value, list) for value in (mappings, events, references)):
        raise ModelGateVerificationError("Gate 1 event-ledger arrays are invalid")
    if not isinstance(exposure, dict):
        raise ModelGateVerificationError("Gate 1 fail-closed exposure is missing")
    mapping_ids = [str(value.get("anchor_episode_id", "")) for value in mappings]
    if not all(mapping_ids) or len(mapping_ids) != len(set(mapping_ids)):
        raise ModelGateVerificationError("Gate 1 oracle mapping identity mismatch")
    for event in [*events, *references]:
        _validate_replacement_event(
            event,
            source_id=source_id,
            source_end=source_end,
            confirmation_samples=confirmation_samples,
        )
    event_ids = [str(value.get("anchor_episode_id", "")) for value in events]
    if len(event_ids) != len(set(event_ids)):
        raise ModelGateVerificationError("Gate 1 emitted duplicate episode events")
    _validate_exposure(exposure)
    if "exclusive_other_contamination_seconds" in exposure:
        _assert_exposure_matches(exposure, _gate1_exposure_replay(row, manifest_row))


def _expected_evaluation_timeline_end(
    manifest_row: dict[str, Any], cfg: dict[str, Any]
) -> int:
    scored_start = int(manifest_row["scored_start_sample"])
    canonical_cells = evaluation_cells(
        intervals_from_manifest(manifest_row),
        scored_start,
        int(manifest_row["scored_end_sample"]),
        int(cfg["evaluation_grid_ms"]) * 16,
    )
    return canonical_cells[-1].end_sample if canonical_cells else scored_start


def _validate_gate2_row(
    row: dict[str, Any], manifest_row: dict[str, Any], cfg: dict[str, Any] | None = None
) -> dict[str, int]:
    confirmation_samples = int(row["replacement_confirm_ms"]) * 16
    source_id = str(row["source_id"])
    source_end = int(manifest_row["source_duration_samples"])
    enrollments = row.get("enrollments")
    episodes = row.get("annotated_episodes")
    timeline = row.get("timeline")
    events = row.get("events")
    references = row.get("reference_events")
    exposure = row.get("fail_closed_exposure")
    if not all(
        isinstance(value, list)
        for value in (enrollments, episodes, timeline, events, references)
    ) or not isinstance(exposure, dict):
        raise ModelGateVerificationError("Gate 2 event-ledger arrays are invalid")
    for event in [*events, *references]:
        _validate_replacement_event(
            event,
            source_id=source_id,
            source_end=source_end,
            confirmation_samples=confirmation_samples,
        )
    event_ids = [str(value.get("anchor_episode_id", "")) for value in events]
    if len(event_ids) != len(set(event_ids)):
        raise ModelGateVerificationError("Gate 2 emitted duplicate episode events")
    for enrollment in enrollments:
        try:
            candidate = int(enrollment["candidate_start_sample"])
            frontier = int(enrollment["model_evidence_frontier_sample"])
            emit = int(enrollment["decoder_emit_sample"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelGateVerificationError("causal enrollment timing is invalid") from exc
        if (
            enrollment.get("source_id") != source_id
            or not 0 <= candidate <= frontier <= emit <= source_end
        ):
            raise ModelGateVerificationError("causal enrollment timing contract mismatch")
    episode_ids: set[str] = set()
    for episode in episodes:
        episode_id = str(episode.get("episode_id", ""))
        try:
            anchor_emit = int(episode["anchor_emit_sample"])
            end_emit = int(episode["end_emit_sample"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelGateVerificationError("causal episode timing is invalid") from exc
        if (
            not episode_id
            or episode_id in episode_ids
            or not 0 <= anchor_emit <= end_emit <= source_end
        ):
            raise ModelGateVerificationError("causal episode timing contract mismatch")
        episode_ids.add(episode_id)
    enrollment_ids = {str(value.get("anchor_episode_id", "")) for value in enrollments}
    if enrollment_ids != episode_ids or len(enrollments) != len(episodes):
        raise ModelGateVerificationError("causal enrollment/episode identity mismatch")
    previous_end: int | None = None
    scored_start = int(manifest_row["scored_start_sample"])
    for span in timeline:
        try:
            start = int(span["start_sample"])
            end = int(span["end_sample"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelGateVerificationError("causal timeline timing is invalid") from exc
        if start < 0 or end <= start or end > source_end or (
            previous_end is not None and start != previous_end
        ):
            raise ModelGateVerificationError("causal timeline is not contiguous")
        if previous_end is None and start != scored_start:
            raise ModelGateVerificationError("causal timeline does not start at scored start")
        previous_end = end
        if span.get("lifecycle") == "ANCHORED":
            anchor_id = span.get("anchor_id")
            matching = [
                episode
                for episode in episodes
                if episode.get("anchor_slot_id") == anchor_id
                and int(episode["anchor_emit_sample"]) <= start
                and end <= int(episode["end_emit_sample"])
            ]
            if not matching:
                raise ModelGateVerificationError(
                    "ANCHORED timeline support precedes its anchor emission"
                )
    expected_timeline_end = _expected_evaluation_timeline_end(
        manifest_row, cfg or config()
    )
    if previous_end != expected_timeline_end:
        raise ModelGateVerificationError(
            "causal timeline does not cover canonical evaluation cells"
        )
    _validate_exposure(exposure)
    if "exclusive_other_contamination_seconds" in exposure:
        _assert_exposure_matches(exposure, _gate2_exposure_replay(row))
    replayed_opportunities = _causal_opportunity_replay(
        row, manifest_row, cfg or config()
    )
    if "expected_opportunities" in row and row["expected_opportunities"] != replayed_opportunities:
        raise ModelGateVerificationError("causal opportunity ledger replay mismatch")
    expected = int(row.get("expected_opportunity_count", -1))
    matched = sum(value.get("opportunity_start_sample") is not None for value in episodes)
    total = len(episodes)
    wrong = sum(value.get("correct_anchor") is not True for value in episodes)
    if expected != len(replayed_opportunities) or matched > expected:
        raise ModelGateVerificationError("causal opportunity denominator is inconsistent")
    return {
        "expected": expected,
        "matched": matched,
        "total": total,
        "unmatched": total - matched,
        "wrong": wrong,
        "failures": expected - matched,
        "slot_losses": sum(
            value.get("end_reason") == "slot_continuity_invalid" for value in episodes
        ),
    }


def _group_event_ledger(
    rows: Sequence[dict[str, Any]],
    *,
    gate: str,
    manifest: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    expected_keys = {
        (gate, family, int(replacement_ms), source_id)
        for family in ("streaming_sortformer", "ls_eend")
        for replacement_ms in cfg["replacement_confirm_ms"]
        for source_id in manifest
    }
    observed_keys: set[tuple[str, str, int, str]] = set()
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        _validate_row_hash(row)
        try:
            key = (
                str(row["gate"]),
                str(row["family"]),
                int(row["replacement_confirm_ms"]),
                str(row["source_id"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelGateVerificationError("event-ledger key is invalid") from exc
        if (
            row.get("schema_version") != "psem.relative_occupancy.gate_event_session.v1"
            or key in observed_keys
            or key[3] not in manifest
        ):
            raise ModelGateVerificationError("event-ledger identity mismatch")
        observed_keys.add(key)
        grouped[key[:3]].append(row)
        if gate == "gate1_oracle_anchor":
            _validate_gate1_row(row, manifest[key[3]])
        else:
            _validate_gate2_row(row, manifest[key[3]], cfg)
    if observed_keys != expected_keys:
        raise ModelGateVerificationError("event-ledger session coverage mismatch")
    for values in grouped.values():
        values.sort(key=lambda value: str(value["source_id"]))
    return grouped


def _topology_window(
    manifest_row: dict[str, Any], episode: dict[str, Any]
) -> tuple[int, int] | None:
    transitions = {value["transition_id"]: value for value in manifest_row["transitions"]}
    indices = [
        int(index)
        for transition_id in episode["transition_ids"]
        if transition_id in transitions
        for index in (
            transitions[transition_id].get("from_interval_index"),
            transitions[transition_id].get("to_interval_index"),
        )
        if index is not None
    ]
    if not indices:
        return None
    intervals = manifest_row["intervals"]
    return (
        int(intervals[min(indices)]["start_sample"]),
        int(intervals[max(indices)]["end_sample"]),
    )


def _independent_topology_slices(
    manifest: dict[str, dict[str, Any]],
    session_rows: Sequence[dict[str, Any]],
    tolerance_samples: int,
) -> dict[str, Any]:
    sessions = {str(row["source_id"]): row for row in session_rows}
    counters: dict[str, dict[str, int]] = {}
    for source_id, manifest_row in sorted(manifest.items()):
        session = sessions[source_id]
        predicted = session["events"]
        references = session["reference_events"]
        for episode in manifest_row["topology_episodes"]:
            if not episode.get("coverage_gate_eligible", False):
                continue
            window = _topology_window(manifest_row, episode)
            if window is None:
                continue
            start, end = window
            topology = str(episode["primary_topology"])
            values = counters.setdefault(
                topology,
                {
                    "eligible_episode_count": 0,
                    "episodes_with_predicted_cut": 0,
                    "episodes_with_reference_replacement": 0,
                    "episodes_with_aligned_cut": 0,
                    "episodes_with_early_cut": 0,
                },
            )
            values["eligible_episode_count"] += 1
            predicted_in_window = [
                value
                for value in predicted
                if start <= int(value["boundary_source_sample"]) < end
            ]
            references_in_window = [
                value
                for value in references
                if start <= int(value["boundary_source_sample"]) < end
            ]
            if predicted_in_window:
                values["episodes_with_predicted_cut"] += 1
            if references_in_window:
                values["episodes_with_reference_replacement"] += 1
            differences = [
                int(left["boundary_source_sample"])
                - int(right["boundary_source_sample"])
                for left in predicted_in_window
                for right in references_in_window
            ]
            if any(0 <= value <= tolerance_samples for value in differences):
                values["episodes_with_aligned_cut"] += 1
            if any(-tolerance_samples <= value < 0 for value in differences):
                values["episodes_with_early_cut"] += 1
    result: dict[str, Any] = {}
    for topology, values in sorted(counters.items()):
        count = values["eligible_episode_count"]
        result[topology] = {
            **values,
            "cut_episode_rate": values["episodes_with_predicted_cut"] / count,
            "aligned_cut_episode_rate": values["episodes_with_aligned_cut"] / count,
            "overlap_return_preservation_rate": (
                1.0 - values["episodes_with_predicted_cut"] / count
                if topology == "overlap_return"
                else None
            ),
            "overlap_takeover_success_rate": (
                values["episodes_with_aligned_cut"] / count
                if topology == "overlap_takeover"
                else None
            ),
        }
    return result


def _validate_aggregate_rows(
    *,
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]],
    product_rows: Sequence[dict[str, Any]],
    topology_rows: Sequence[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    tolerance_samples: int,
) -> None:
    products = {
        (str(row.get("gate")), str(row.get("family")), int(row["replacement_confirm_ms"])): row
        for row in product_rows
        if row.get("gate") in {"gate1_oracle_anchor", "gate2_causal_anchor"}
    }
    topologies = {
        (str(row.get("gate")), str(row.get("family")), int(row["replacement_confirm_ms"])): row
        for row in topology_rows
    }
    for key, session_rows in grouped.items():
        product = products.get(key)
        topology = topologies.get(key)
        if product is None or topology is None:
            raise ModelGateVerificationError("event ledger lacks an aggregate artifact row")
        predicted_count = sum(len(row["events"]) for row in session_rows)
        reference_count = sum(len(row["reference_events"]) for row in session_rows)
        if (
            product.get("source_count") != len(session_rows)
            or product.get("predicted_cut_count") != predicted_count
            or product.get("reference_replacement_count") != reference_count
        ):
            raise ModelGateVerificationError("event-ledger aggregate count mismatch")
        for field in (
            "masked_seconds",
            "masked_active_speech_seconds",
            "unanchored_active_speech_seconds",
            "anchor_uncertain_active_speech_seconds",
            "fail_closed_unknown_active_speech_seconds",
            "exclusive_other_contamination_upper_bound_seconds",
        ):
            expected = sum(float(row["fail_closed_exposure"][field]) for row in session_rows)
            if abs(float(product[field]) - expected) > 1e-9:
                raise ModelGateVerificationError(
                    f"event-ledger fail-closed aggregate mismatch: {field}"
                )
        if float(product["exclusive_other_contamination_upper_bound_seconds"]) < float(
            product["exclusive_other_contamination_seconds"]
        ):
            raise ModelGateVerificationError("contamination upper bound is not conservative")
        expected_slices = _independent_topology_slices(
            manifest, session_rows, tolerance_samples
        )
        if topology.get("slices") != expected_slices:
            raise ModelGateVerificationError("independent topology slice replay mismatch")
        if key[0] == "gate2_causal_anchor":
            derived = [
                _validate_gate2_row(row, manifest[str(row["source_id"])])
                for row in session_rows
            ]
            expected_counts = {
                "expected_opportunity_count": sum(value["expected"] for value in derived),
                "enrollment_count": sum(value["matched"] for value in derived),
                "total_enrollment_count": sum(value["total"] for value in derived),
                "unmatched_enrollment_count": sum(value["unmatched"] for value in derived),
                "wrong_anchor_count": sum(value["wrong"] for value in derived),
                "enrollment_failure_count": sum(value["failures"] for value in derived),
                "slot_loss_count": sum(value["slot_losses"] for value in derived),
            }
            if any(product.get(field) != value for field, value in expected_counts.items()):
                raise ModelGateVerificationError(
                    "causal opportunity/enrollment aggregate mismatch"
                )


def _validate_semantics(
    artifacts: dict[str, dict[str, Any]],
    gate1_rows: list[dict[str, Any]],
    gate2_rows: list[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
) -> None:
    gate1 = artifacts["gate1"]
    gate2 = artifacts["gate2"]
    product = artifacts["product"]
    topology = artifacts["topology"]
    latency = artifacts["latency"]
    selection = artifacts["selection"]
    cfg = config()
    expected_families = {"streaming_sortformer", "ls_eend"}
    if set(gate1.get("families", {})) != expected_families:
        raise ModelGateVerificationError("Gate 1 family coverage mismatch")
    if set(gate2.get("families", {})) != expected_families:
        raise ModelGateVerificationError("Gate 2 family coverage mismatch")
    expected_grid = {
        (float(active), float(low), int(confirm_ms) * 16)
        for active in cfg["causal_enrollment"]["active_thresholds"]
        for low in cfg["causal_enrollment"]["other_low_thresholds"]
        if float(low) < float(active)
        for confirm_ms in cfg["causal_enrollment"]["confirm_ms"]
    }
    if (
        cfg["causal_enrollment"].get("validity_rule")
        != "other_low_threshold < active_threshold"
        or len(expected_grid) != int(cfg["causal_enrollment"]["valid_candidate_count"])
        or selection.get("causal_enrollment_grid") != cfg["causal_enrollment"]
    ):
        raise ModelGateVerificationError("causal selection grid declaration mismatch")
    for family in expected_families:
        gate1_family = gate1["families"][family]
        points = gate1_family["primitive"]["operating_points"]
        expected_point = min(
            points,
            key=lambda value: (
                -float(value["macro_f1"]),
                float(value["anchor_threshold"]),
                float(value["other_threshold"]),
            ),
        )
        selected_point = gate1_family["primitive"]["selected_operating_point"]
        if any(
            selected_point[field] != expected_point[field]
            for field in ("anchor_threshold", "other_threshold", "macro_f1")
        ):
            raise ModelGateVerificationError(f"Gate 1 selection mismatch: {family}")
        gate2_family = gate2["families"][family]
        grid = gate2_family["selection_grid"]
        observed_grid = {
            (
                float(row["config"]["active_threshold"]),
                float(row["config"]["other_low_threshold"]),
                int(row["config"]["confirmation_samples"]),
            )
            for row in grid
        }
        if observed_grid != expected_grid or len(grid) != len(expected_grid):
            raise ModelGateVerificationError(f"Gate 2 grid coverage mismatch: {family}")
        for row in grid:
            metrics = row["metrics"]
            if (
                int(metrics["enrollment_count"])
                > int(metrics["expected_opportunity_count"])
                or int(metrics["unmatched_enrollment_count"])
                != int(metrics["total_enrollment_count"])
                - int(metrics["enrollment_count"])
            ):
                raise ModelGateVerificationError(
                    f"Gate 2 selection denominator mismatch: {family}"
                )
        expected_config = min(grid, key=_selection_key)
        if gate2_family["selected_configuration"] != expected_config:
            raise ModelGateVerificationError(f"Gate 2 selection mismatch: {family}")
        if gate1_family["model_receipt"] != gate2_family["model_receipt"]:
            raise ModelGateVerificationError(f"Gate 1/2 trace identity mismatch: {family}")
    rows = product.get("rows", [])
    gates = [str(value.get("gate")) for value in rows]
    expected_counts = {
        "vad_only_no_speaker_cut": 1,
        "gate0_oracle": 4,
        "gate1_oracle_anchor": 8,
        "gate2_causal_anchor": 8,
    }
    if {key: gates.count(key) for key in expected_counts} != expected_counts:
        raise ModelGateVerificationError("product baseline/frontier coverage mismatch")
    if not topology.get("rows") or len(topology["rows"]) != 16:
        raise ModelGateVerificationError("topology slice coverage mismatch")
    if set(latency.get("families", {})) != expected_families:
        raise ModelGateVerificationError("latency family coverage mismatch")
    payload = dict(selection)
    observed_hash = payload.pop("selection_sha256", None)
    if observed_hash != canonical_sha256(payload):
        raise ModelGateVerificationError("selection payload hash mismatch")
    if (
        selection.get("eval_open_authorized") is not False
        or selection.get("eval_open_count") != 0
        or selection.get("eval_status") != "sealed"
        or selection.get("same_cached_trace_for_gate1_and_gate2") is not True
    ):
        raise ModelGateVerificationError("DEV selection did not preserve the EVAL seal")
    tolerance_samples = int(cfg["product_event_alignment_tolerance_ms"]) * 16
    gate1_grouped = _group_event_ledger(
        gate1_rows,
        gate="gate1_oracle_anchor",
        manifest=manifest,
        cfg=cfg,
    )
    gate2_grouped = _group_event_ledger(
        gate2_rows,
        gate="gate2_causal_anchor",
        manifest=manifest,
        cfg=cfg,
    )
    _validate_aggregate_rows(
        grouped=gate1_grouped,
        product_rows=artifacts["gate1_product"]["rows"],
        topology_rows=artifacts["gate1_topology"]["rows"],
        manifest=manifest,
        tolerance_samples=tolerance_samples,
    )
    _validate_aggregate_rows(
        grouped=gate2_grouped,
        product_rows=product["rows"],
        topology_rows=topology["rows"],
        manifest=manifest,
        tolerance_samples=tolerance_samples,
    )


def _validate_artifact_bindings(
    paths: dict[str, Path], artifacts: dict[str, dict[str, Any]]
) -> None:
    gate0 = artifacts["gate0"]
    gate0_verification = artifacts["gate0_verification"]
    if (
        gate0.get("schema_version") != "psem.relative_occupancy.gate0_metrics.v1"
        or gate0.get("passed") is not True
        or gate0_verification.get("schema_version")
        != "psem.relative_occupancy.gate0_verification.v1"
        or gate0_verification.get("passed") is not True
        or gate0_verification.get("metrics_sha256") != sha256_file(paths["gate0"])
    ):
        raise ModelGateVerificationError("Gate 0 accepted verification binding mismatch")
    selection = artifacts["selection"]
    bindings = selection.get("artifact_bindings")
    expected = {
        "gate0_metrics_sha256": "gate0",
        "gate0_verification_sha256": "gate0_verification",
        "gate1_metrics_sha256": "gate1",
        "gate1_product_frontier_sha256": "gate1_product",
        "gate1_topology_slices_sha256": "gate1_topology",
        "gate1_latency_breakdown_sha256": "gate1_latency",
        "gate1_event_ledger_sha256": "gate1_events",
        "gate2_metrics_sha256": "gate2",
        "gate2_event_ledger_sha256": "gate2_events",
        "product_frontiers_sha256": "product",
        "topology_slices_sha256": "topology",
        "latency_breakdown_sha256": "latency",
    }
    if not isinstance(bindings, dict) or any(
        bindings.get(field) != sha256_file(paths[name]) for field, name in expected.items()
    ):
        raise ModelGateVerificationError("DEV selection artifact hash binding mismatch")
    if (
        artifacts["gate1"].get("event_ledger_sha256")
        != sha256_file(paths["gate1_events"])
        or artifacts["gate2"].get("event_ledger_sha256")
        != sha256_file(paths["gate2_events"])
        or artifacts["gate2"].get("gate0_verification_sha256")
        != sha256_file(paths["gate0_verification"])
    ):
        raise ModelGateVerificationError("gate artifact supporting hash mismatch")


def run(args: argparse.Namespace) -> None:
    paths = {
        "gate0": Path(args.gate0).resolve(),
        "gate0_verification": Path(args.gate0_verification).resolve(),
        "gate1": Path(args.gate1).resolve(),
        "gate1_product": Path(args.gate1_product).resolve(),
        "gate1_topology": Path(args.gate1_topology).resolve(),
        "gate1_latency": Path(args.gate1_latency).resolve(),
        "gate1_events": Path(args.gate1_events).resolve(),
        "gate2": Path(args.gate2).resolve(),
        "gate2_events": Path(args.gate2_events).resolve(),
        "product": Path(args.product).resolve(),
        "topology": Path(args.topology).resolve(),
        "latency": Path(args.latency).resolve(),
        "selection": Path(args.selection).resolve(),
    }
    output = safe_output_path(Path(args.output))
    protected_inputs = {
        *paths.values(),
        Path(args.manifest).resolve(),
        Path(args.sortformer_receipt).resolve(),
        Path(args.lseend_receipt).resolve(),
    }
    if output in protected_inputs:
        raise ModelGateVerificationError("verification output aliases an input")
    object_names = set(paths) - {"gate1_events", "gate2_events"}
    artifacts = {name: _load_object(paths[name]) for name in object_names}
    gate1_rows = load_jsonl(paths["gate1_events"])
    gate2_rows = load_jsonl(paths["gate2_events"])
    manifest = _validate_manifest(Path(args.manifest).resolve())
    _validate_artifact_bindings(paths, artifacts)
    _validate_semantics(artifacts, gate1_rows, gate2_rows, manifest)
    with tempfile.TemporaryDirectory(prefix="psem-issue97-verify-") as temporary:
        root = Path(temporary)
        regenerated = {
            name: root / paths[name].name
            for name in paths
            if name not in {"gate0", "gate0_verification"}
        }
        run_gate1(
            Namespace(
                manifest=args.manifest,
                sortformer_receipt=args.sortformer_receipt,
                lseend_receipt=args.lseend_receipt,
                output=str(regenerated["gate1"]),
                product_output=str(regenerated["gate1_product"]),
                topology_output=str(regenerated["gate1_topology"]),
                latency_output=str(regenerated["gate1_latency"]),
                event_output=str(regenerated["gate1_events"]),
            )
        )
        run_gate2(
            Namespace(
                manifest=args.manifest,
                sortformer_receipt=args.sortformer_receipt,
                lseend_receipt=args.lseend_receipt,
                gate0=args.gate0,
                gate0_verification=args.gate0_verification,
                gate1=str(regenerated["gate1"]),
                gate1_events=str(regenerated["gate1_events"]),
                gate1_product=str(regenerated["gate1_product"]),
                gate1_topology=str(regenerated["gate1_topology"]),
                latency=str(regenerated["gate1_latency"]),
                output=str(regenerated["gate2"]),
                product_output=str(regenerated["product"]),
                topology_output=str(regenerated["topology"]),
                latency_output=str(regenerated["latency"]),
                selection_output=str(regenerated["selection"]),
                event_output=str(regenerated["gate2_events"]),
            )
        )
        for name, regenerated_path in regenerated.items():
            if name.endswith("events"):
                equal = load_jsonl(paths[name]) == load_jsonl(regenerated_path)
            else:
                equal = _load_object(paths[name]) == _load_object(regenerated_path)
            if not equal:
                raise ModelGateVerificationError(
                    f"deterministic regeneration mismatch: {name}"
                )
    receipt = {
        "schema_version": "psem.relative_occupancy.model_gate_verification.v1",
        "role": "PSEM-STRATEGY-DEV",
        "eval_status": "sealed",
        "passed": True,
        "deterministic_regeneration": True,
        "independent_structural_verification": True,
        "artifact_sha256": {
            name: sha256_file(path) for name, path in sorted(paths.items())
        },
        "selection_sha256": artifacts["selection"]["selection_sha256"],
    }
    write_json(output, receipt)
    print({"output": str(output), "passed": True})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sortformer-receipt", required=True)
    parser.add_argument("--lseend-receipt", required=True)
    parser.add_argument("--gate0", required=True)
    parser.add_argument("--gate0-verification", required=True)
    parser.add_argument("--gate1", required=True)
    parser.add_argument("--gate1-product", required=True)
    parser.add_argument("--gate1-topology", required=True)
    parser.add_argument("--gate1-latency", required=True)
    parser.add_argument("--gate1-events", required=True)
    parser.add_argument("--gate2", required=True)
    parser.add_argument("--gate2-events", required=True)
    parser.add_argument("--product", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--latency", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
