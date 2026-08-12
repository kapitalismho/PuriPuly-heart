from __future__ import annotations

import hashlib
import json
import statistics
from bisect import bisect_left
from collections import deque
from collections.abc import Iterable, Sequence
from heapq import heappop, heappush
from typing import Any

SAMPLE_RATE = 16000
KIND_PRIORITY = {
    "overlap_onset": 0,
    "dominant_replacement": 1,
    "new_track_onset": 2,
    "track_instability": 3,
    "speaker_change_unknown": 4,
}
HARD_KINDS = {"dominant_replacement", "speaker_change_unknown"}
SOFT_KINDS = {"overlap_onset"}


class Phase5PolicyError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def samples(milliseconds: int) -> int:
    return milliseconds * SAMPLE_RATE // 1000


def proposal_order(row: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        int(row["observed_source_sample_at_emit"]),
        int(row["boundary_source_sample"]),
        str(row["profile_id"]),
        str(row["proposal_id"]),
    )


def validate_proposals(proposals: Sequence[dict[str, Any]]) -> None:
    seen: set[str] = set()
    identities: set[tuple[str, str, str, int, str]] = set()
    for proposal in proposals:
        proposal_id = str(proposal["proposal_id"])
        if proposal_id in seen:
            raise Phase5PolicyError(f"duplicate proposal_id: {proposal_id}")
        seen.add(proposal_id)
        if int(proposal["observed_source_sample_at_emit"]) < int(
            proposal["boundary_source_sample"]
        ):
            raise Phase5PolicyError(f"anticipatory proposal: {proposal_id}")
        if "audio_epoch" not in proposal or "source_session_id" not in proposal:
            raise Phase5PolicyError(f"proposal identity incomplete: {proposal_id}")
        if not proposal.get("confidence_semantics_id"):
            raise Phase5PolicyError(f"confidence semantics missing: {proposal_id}")
        identities.add(
            (
                str(proposal["family"]),
                str(proposal["checkpoint"]),
                str(proposal["profile_id"]),
                int(proposal["audio_epoch"]),
                str(proposal["source_session_id"]),
            )
        )
    if len(identities) > 1:
        raise Phase5PolicyError("one replay must contain one profile, epoch, and source")


def output_kind(members: Sequence[dict[str, Any]]) -> str:
    kinds = {str(row["proposal_kind"]) for row in members}
    if kinds == {"speaker_change_unknown"}:
        return "speaker_change_unknown"
    return min(kinds, key=lambda kind: (KIND_PRIORITY.get(kind, 999), kind))


def compatible_members(members: Sequence[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [row for row in members if str(row["proposal_kind"]) == kind]


def first_representative(members: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return min(members, key=proposal_order)


def max_confidence_representative(
    members: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    semantics = {str(row["confidence_semantics_id"]) for row in members}
    if len(semantics) != 1:
        return first_representative(members), "first_incompatible_confidence_semantics"
    boundaries = [int(row["boundary_source_sample"]) for row in members]
    median = float(statistics.median(boundaries))
    selected = min(
        members,
        key=lambda row: (
            -float(row["confidence"]),
            int(row["observed_source_sample_at_emit"]),
            abs(int(row["boundary_source_sample"]) - median),
            int(row["boundary_source_sample"]),
            str(row["proposal_id"]),
        ),
    )
    return selected, "max_confidence"


def cluster_proposals_reference(
    proposals: Sequence[dict[str, Any]],
    *,
    cluster_debounce_ms: int,
    cluster_boundary_radius_ms: int,
    refractory_ms: int,
    representative: str,
    episode_observed_end: int,
) -> dict[str, Any]:
    validate_proposals(proposals)
    if representative not in ("first", "max_confidence"):
        raise Phase5PolicyError(f"unsupported representative: {representative}")
    debounce = samples(cluster_debounce_ms)
    radius = samples(cluster_boundary_radius_ms)
    refractory = samples(refractory_ms)
    pending = sorted((dict(row) for row in proposals), key=proposal_order)
    if (
        pending
        and max(int(row["observed_source_sample_at_emit"]) for row in pending)
        > episode_observed_end
    ):
        raise Phase5PolicyError("proposal observation exceeds episode end")
    clusters: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    ready_frontier_by_epoch: dict[int, int] = {}
    while pending:
        first = pending.pop(0)
        epoch = int(first["audio_epoch"])
        opened = max(
            int(first["observed_source_sample_at_emit"]),
            ready_frontier_by_epoch.get(epoch, 0),
        )
        anchor = int(first["boundary_source_sample"])
        scheduled_close = opened + debounce
        close = min(scheduled_close, episode_observed_end)
        members = [first]
        queued: list[dict[str, Any]] = []
        for proposal in pending:
            joins = (
                int(proposal["audio_epoch"]) == epoch
                and int(proposal["observed_source_sample_at_emit"]) <= close
                and abs(int(proposal["boundary_source_sample"]) - anchor) <= radius
            )
            if joins:
                members.append(proposal)
            else:
                queued.append(proposal)
        kind = output_kind(members)
        compatible = compatible_members(members, kind)
        if representative == "first":
            selected = first_representative(compatible)
            reason = "first"
        else:
            selected, reason = max_confidence_representative(compatible)
        availability = max(close, int(selected["observed_source_sample_at_emit"]))
        cluster_id = (
            "cluster:"
            + content_sha256(
                {
                    "member_ids": sorted(str(row["proposal_id"]) for row in members),
                    "kind": kind,
                    "representative_id": selected["proposal_id"],
                    "availability": availability,
                }
            )[:24]
        )
        clusters.append(
            {
                "cluster_id": cluster_id,
                "audio_epoch": epoch,
                "source_session_id": str(selected["source_session_id"]),
                "proposal_kind": kind,
                "member_proposal_ids": sorted(str(row["proposal_id"]) for row in members),
                "compatible_representative_subset": sorted(
                    str(row["proposal_id"]) for row in compatible
                ),
                "representative_proposal_id": str(selected["proposal_id"]),
                "representative_reason": reason,
                "confidence_semantics_id": str(selected["confidence_semantics_id"]),
                "confidence": float(selected["confidence"]),
                "boundary_source_sample": int(selected["boundary_source_sample"]),
                "observed_source_sample_at_emit": availability,
                "cluster_open_frontier": opened,
                "cluster_close_frontier": close,
                "scheduled_close_frontier": scheduled_close,
                "tail_closed": close < scheduled_close,
                "boundary_spread_samples": max(
                    int(row["boundary_source_sample"]) for row in members
                )
                - min(int(row["boundary_source_sample"]) for row in members),
                "confidence_distribution": sorted(float(row["confidence"]) for row in compatible),
            }
        )
        cutoff = availability + refractory
        ready_frontier_by_epoch[epoch] = availability
        next_pending: list[dict[str, Any]] = []
        for proposal in queued:
            proposal_frontier = max(
                int(proposal["observed_source_sample_at_emit"]),
                ready_frontier_by_epoch.get(int(proposal["audio_epoch"]), 0),
            )
            if int(proposal["audio_epoch"]) == epoch and proposal_frontier < cutoff:
                suppressed.append(
                    {
                        "proposal_id": str(proposal["proposal_id"]),
                        "owner_cluster_id": cluster_id,
                        "reason": "refractory",
                        "refractory_until_sample": cutoff,
                    }
                )
            else:
                next_pending.append(proposal)
        pending = next_pending
    return {
        "clusters": clusters,
        "refractory_proposals": suppressed,
        "cluster_count": len(clusters),
        "refractory_count": len(suppressed),
    }


def cluster_proposals(
    proposals: Sequence[dict[str, Any]],
    *,
    cluster_debounce_ms: int,
    cluster_boundary_radius_ms: int,
    refractory_ms: int,
    representative: str,
    episode_observed_end: int,
) -> dict[str, Any]:
    validate_proposals(proposals)
    if representative not in ("first", "max_confidence"):
        raise Phase5PolicyError(f"unsupported representative: {representative}")
    debounce = samples(cluster_debounce_ms)
    radius = samples(cluster_boundary_radius_ms)
    refractory = samples(refractory_ms)
    pending = deque(sorted((dict(row) for row in proposals), key=proposal_order))
    if (
        pending
        and max(int(row["observed_source_sample_at_emit"]) for row in pending)
        > episode_observed_end
    ):
        raise Phase5PolicyError("proposal observation exceeds episode end")
    clusters: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    ready_frontier_by_epoch: dict[int, int] = {}
    while pending:
        first = pending.popleft()
        epoch = int(first["audio_epoch"])
        opened = max(
            int(first["observed_source_sample_at_emit"]),
            ready_frontier_by_epoch.get(epoch, 0),
        )
        anchor = int(first["boundary_source_sample"])
        scheduled_close = opened + debounce
        close = min(scheduled_close, episode_observed_end)
        members = [first]
        queued: list[dict[str, Any]] = []
        while pending and int(pending[0]["observed_source_sample_at_emit"]) <= close:
            candidate = pending.popleft()
            if (
                int(candidate["audio_epoch"]) == epoch
                and abs(int(candidate["boundary_source_sample"]) - anchor) <= radius
            ):
                members.append(candidate)
            else:
                queued.append(candidate)
        kind = output_kind(members)
        compatible = compatible_members(members, kind)
        if representative == "first":
            selected = first_representative(compatible)
            reason = "first"
        else:
            selected, reason = max_confidence_representative(compatible)
        availability = max(close, int(selected["observed_source_sample_at_emit"]))
        cluster_id = (
            "cluster:"
            + content_sha256(
                {
                    "member_ids": sorted(str(row["proposal_id"]) for row in members),
                    "kind": kind,
                    "representative_id": selected["proposal_id"],
                    "availability": availability,
                }
            )[:24]
        )
        clusters.append(
            {
                "cluster_id": cluster_id,
                "audio_epoch": epoch,
                "source_session_id": str(selected["source_session_id"]),
                "proposal_kind": kind,
                "member_proposal_ids": sorted(str(row["proposal_id"]) for row in members),
                "compatible_representative_subset": sorted(
                    str(row["proposal_id"]) for row in compatible
                ),
                "representative_proposal_id": str(selected["proposal_id"]),
                "representative_reason": reason,
                "confidence_semantics_id": str(selected["confidence_semantics_id"]),
                "confidence": float(selected["confidence"]),
                "boundary_source_sample": int(selected["boundary_source_sample"]),
                "observed_source_sample_at_emit": availability,
                "cluster_open_frontier": opened,
                "cluster_close_frontier": close,
                "scheduled_close_frontier": scheduled_close,
                "tail_closed": close < scheduled_close,
                "boundary_spread_samples": max(
                    int(row["boundary_source_sample"]) for row in members
                )
                - min(int(row["boundary_source_sample"]) for row in members),
                "confidence_distribution": sorted(float(row["confidence"]) for row in compatible),
            }
        )
        cutoff = availability + refractory
        ready_frontier_by_epoch[epoch] = availability
        if refractory:
            for candidate in queued:
                suppressed.append(
                    {
                        "proposal_id": str(candidate["proposal_id"]),
                        "owner_cluster_id": cluster_id,
                        "reason": "refractory",
                        "refractory_until_sample": cutoff,
                    }
                )
            while pending:
                candidate = pending[0]
                proposal_frontier = max(
                    int(candidate["observed_source_sample_at_emit"]),
                    ready_frontier_by_epoch.get(int(candidate["audio_epoch"]), 0),
                )
                if int(candidate["audio_epoch"]) != epoch or proposal_frontier >= cutoff:
                    break
                candidate = pending.popleft()
                suppressed.append(
                    {
                        "proposal_id": str(candidate["proposal_id"]),
                        "owner_cluster_id": cluster_id,
                        "reason": "refractory",
                        "refractory_until_sample": cutoff,
                    }
                )
        else:
            pending.extendleft(reversed(queued))
    return {
        "clusters": clusters,
        "refractory_proposals": suppressed,
        "cluster_count": len(clusters),
        "refractory_count": len(suppressed),
    }


def actionize_clusters(clusters: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for cluster in clusters:
        kind = str(cluster["proposal_kind"])
        if kind in HARD_KINDS:
            requested = "hard_candidate"
        elif kind in SOFT_KINDS:
            requested = "soft_marker"
        else:
            requested = "diagnostic_only"
        actions.append(
            {
                "detector_action_id": f"detector:{cluster['cluster_id']}",
                "cluster_id": str(cluster["cluster_id"]),
                "audio_epoch": int(cluster["audio_epoch"]),
                "source_session_id": str(cluster["source_session_id"]),
                "proposal_kind": kind,
                "requested_action": requested,
                "boundary_source_sample": int(cluster["boundary_source_sample"]),
                "observed_source_sample_at_emit": int(cluster["observed_source_sample_at_emit"]),
                "confidence": float(cluster["confidence"]),
                "confidence_semantics_id": str(cluster["confidence_semantics_id"]),
            }
        )
    return actions


def lifecycle_silence_intervals(
    lifecycle_events: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in lifecycle_events),
        key=lambda row: (
            int(row["observed_source_sample_at_emit"]),
            int(row["event_source_sample"]),
            str(row["event_id"]),
        ),
    )
    identities = {(int(row["audio_epoch"]), str(row["source_session_id"])) for row in ordered}
    if len(identities) > 1:
        raise Phase5PolicyError("lifecycle replay must contain one source and epoch")
    pending_end: dict[str, Any] | None = None
    intervals: list[dict[str, Any]] = []
    for event in ordered:
        kind = str(event["event_kind"])
        if kind == "speech_end":
            pending_end = event
        elif kind == "speech_start" and pending_end is not None:
            start = int(pending_end["event_source_sample"])
            end = int(event["event_source_sample"])
            if end < start:
                raise Phase5PolicyError("lifecycle silence interval is negative")
            identity = {
                "audio_epoch": int(event["audio_epoch"]),
                "source_session_id": str(event["source_session_id"]),
                "speech_end_event_id": str(pending_end["event_id"]),
                "speech_start_event_id": str(event["event_id"]),
                "start": start,
                "end": end,
            }
            intervals.append(
                {
                    **identity,
                    "silence_interval_id": "silence:" + content_sha256(identity)[:24],
                    "known_at_source_sample": int(event["observed_source_sample_at_emit"]),
                    "speech_end_reason": str(pending_end.get("reason", "")),
                }
            )
            pending_end = None
    return intervals


def projected_silence_intervals(
    vad_actions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    for raw in vad_actions:
        debug = raw.get("debug") if isinstance(raw.get("debug"), dict) else {}
        start_value = raw.get("prev_speech_end_sample", debug.get("prev_speech_end_sample"))
        reason = str(raw.get("prev_end_reason", debug.get("prev_end_reason", "")))
        if start_value is None or reason != "silence":
            continue
        start = int(start_value)
        end = int(raw["boundary_source_sample"])
        if end < start:
            raise Phase5PolicyError("projected silence interval is negative")
        action_id = str(raw.get("action_id", raw.get("event_id", "")))
        identity = {
            "audio_epoch": int(raw["audio_epoch"]),
            "source_session_id": str(raw["source_session_id"]),
            "vad_action_id": action_id,
            "start": start,
            "end": end,
        }
        intervals.append(
            {
                **identity,
                "silence_interval_id": "silence:" + content_sha256(identity)[:24],
                "known_at_source_sample": int(raw["observed_source_sample_at_emit"]),
                "speech_end_reason": reason,
                "projection_source": "pinned_b0_vad_debug",
            }
        )
    return intervals


def derive_fusion_context(
    vad_actions: Sequence[dict[str, Any]],
    detector_actions: Sequence[dict[str, Any]],
    lifecycle_events: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    intervals = lifecycle_silence_intervals(lifecycle_events)
    interval_coordinates = {
        (
            int(row["audio_epoch"]),
            str(row["source_session_id"]),
            int(row["start"]),
            int(row["end"]),
        )
        for row in intervals
    }
    for row in projected_silence_intervals(vad_actions):
        coordinates = (
            int(row["audio_epoch"]),
            str(row["source_session_id"]),
            int(row["start"]),
            int(row["end"]),
        )
        if coordinates not in interval_coordinates:
            intervals.append(row)
            interval_coordinates.add(coordinates)
    vad_rows: list[dict[str, Any]] = []
    for raw in vad_actions:
        row = dict(raw)
        if str(row.get("action_kind", "")) == "structural_max_duration":
            row["association_forbidden"] = "structural_max_duration"
        candidates = [
            interval
            for interval in intervals
            if int(row["audio_epoch"]) == int(interval["audio_epoch"])
            and str(row["source_session_id"]) == str(interval["source_session_id"])
            and int(row["boundary_source_sample"]) == int(interval["end"])
            and int(row["observed_source_sample_at_emit"])
            >= int(interval["known_at_source_sample"])
        ]
        if len(candidates) > 1:
            raise Phase5PolicyError("VAD action maps to multiple silence intervals")
        if candidates:
            interval = candidates[0]
            row["silence_interval_id"] = str(interval["silence_interval_id"])
            row["preceding_silence_start_sample"] = int(interval["start"])
            row["preceding_silence_end_sample"] = int(interval["end"])
            row["silence_interval_known_at_sample"] = int(interval["known_at_source_sample"])
        vad_rows.append(row)
    detector_rows: list[dict[str, Any]] = []
    for raw in detector_actions:
        row = dict(raw)
        candidates = [
            interval
            for interval in intervals
            if int(row["audio_epoch"]) == int(interval["audio_epoch"])
            and str(row["source_session_id"]) == str(interval["source_session_id"])
            and int(interval["start"]) <= int(row["boundary_source_sample"]) <= int(interval["end"])
        ]
        if len(candidates) > 1:
            raise Phase5PolicyError("detector action maps to multiple silence intervals")
        if candidates:
            interval = candidates[0]
            row["silence_interval_id"] = str(interval["silence_interval_id"])
            row["silence_interval_known_at_sample"] = int(interval["known_at_source_sample"])
        detector_rows.append(row)
    return vad_rows, detector_rows, intervals


def action_order(row: dict[str, Any]) -> tuple[int, int, int, str]:
    source_type = 0 if row["origin"] == "vad" else 1
    return (
        int(row["observed_source_sample_at_emit"]),
        int(row["boundary_source_sample"]),
        source_type,
        str(row["event_id"]),
    )


def association_forbidden_reason(left: dict[str, Any], right: dict[str, Any]) -> str | None:
    if left.get("association_forbidden") or right.get("association_forbidden"):
        return str(left.get("association_forbidden") or right.get("association_forbidden"))
    if int(left["audio_epoch"]) != int(right["audio_epoch"]):
        return "cross_epoch"
    if str(left["source_session_id"]) != str(right["source_session_id"]):
        return "cross_source"
    detector = left if left.get("origin") == "detector" else right
    vad = left if left.get("origin") == "vad" else right
    if (
        detector.get("origin") == "detector"
        and vad.get("origin") == "vad"
        and int(vad["observed_source_sample_at_emit"])
        > int(detector["observed_source_sample_at_emit"])
        and vad.get("preceding_silence_start_sample") is not None
        and int(detector["boundary_source_sample"]) < int(vad["preceding_silence_start_sample"])
        and detector.get("silence_interval_id") != vad.get("silence_interval_id")
    ):
        return "vad_ends_post_detector_turn"
    return None


def association_decision(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    radius_samples: int,
    same_silence_interval_association: bool,
) -> tuple[bool, str]:
    forbidden = association_forbidden_reason(left, right)
    if forbidden is not None:
        return False, forbidden
    if (
        abs(int(left["boundary_source_sample"]) - int(right["boundary_source_sample"]))
        <= radius_samples
    ):
        return True, "radius"
    left_interval = left.get("silence_interval_id")
    right_interval = right.get("silence_interval_id")
    associated = bool(
        same_silence_interval_association
        and left_interval is not None
        and left_interval == right_interval
    )
    return associated, "same_silence_interval" if associated else "outside_radius"


def associates(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    radius_samples: int,
    same_silence_interval_association: bool,
) -> bool:
    return association_decision(
        left,
        right,
        radius_samples=radius_samples,
        same_silence_interval_association=same_silence_interval_association,
    )[0]


def fuse_actions(
    vad_actions: Sequence[dict[str, Any]],
    detector_actions: Sequence[dict[str, Any]],
    *,
    detector_vad_radius_ms: int,
    same_silence_interval_association: bool,
) -> dict[str, Any]:
    radius = samples(detector_vad_radius_ms)
    events: list[dict[str, Any]] = []
    for row in vad_actions:
        events.append(
            {
                **row,
                "origin": "vad",
                "event_id": str(row["action_id"]),
                "requested_action": "hard_candidate",
            }
        )
    for row in detector_actions:
        events.append(
            {
                **row,
                "origin": "detector",
                "event_id": str(row["detector_action_id"]),
            }
        )
    events.sort(key=action_order)
    final: list[dict[str, Any]] = []
    final_vad: list[dict[str, Any]] = []
    unassociated_detector: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []

    def associated_candidate(
        candidates: Iterable[dict[str, Any]], event: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str | None]:
        for candidate in candidates:
            associated, reason = association_decision(
                candidate,
                event,
                radius_samples=radius,
                same_silence_interval_association=same_silence_interval_association,
            )
            if associated:
                return candidate, reason
            if reason not in ("outside_radius", "cross_epoch", "cross_source"):
                evidence.append(
                    {
                        "event_id": str(event["event_id"]),
                        "candidate_event_id": str(candidate["event_id"]),
                        "action_kind": "association_rejected",
                        "reason": reason,
                    }
                )
        return None, None

    for event in events:
        requested = str(event["requested_action"])
        if requested == "diagnostic_only":
            evidence.append(
                {
                    "event_id": event["event_id"],
                    "action_kind": "unscored_action",
                    "reason": "diagnostic_only",
                }
            )
            continue
        if requested == "soft_marker":
            final.append(
                {
                    **event,
                    "action_kind": "emit_soft_marker",
                    "final_action_id": f"final:{event['event_id']}",
                }
            )
            continue
        if event["origin"] == "detector":
            prior_vad, association_basis = associated_candidate(
                reversed(final_vad),
                event,
            )
            if prior_vad is not None:
                evidence.append(
                    {
                        "event_id": event["event_id"],
                        "action_kind": "suppress_detector_duplicate",
                        "owner_final_action_id": prior_vad["final_action_id"],
                        "association_basis": association_basis,
                    }
                )
                continue
            created = {
                **event,
                "action_kind": "add_hard_boundary",
                "final_action_id": f"final:{event['event_id']}",
            }
            final.append(created)
            unassociated_detector.append(created)
            continue
        prior_detector, association_basis = associated_candidate(
            reversed(unassociated_detector),
            event,
        )
        if prior_detector is not None:
            prior_detector["action_kind"] = "accelerate_or_replace_vad"
            prior_detector["associated_vad_action_id"] = event["event_id"]
            unassociated_detector.remove(prior_detector)
            evidence.append(
                {
                    "event_id": event["event_id"],
                    "action_kind": "suppress_vad_duplicate",
                    "owner_final_action_id": prior_detector["final_action_id"],
                    "association_basis": association_basis,
                }
            )
            continue
        created = {
            **event,
            "action_kind": str(event.get("action_kind", "retain_vad")),
            "final_action_id": f"final:{event['event_id']}",
        }
        final.append(created)
        final_vad.append(created)
    hard_positions = [
        int(row["boundary_source_sample"])
        for row in final
        if row["action_kind"] in ("retain_vad", "accelerate_or_replace_vad", "add_hard_boundary")
    ]
    if len(hard_positions) != len(set(hard_positions)):
        raise Phase5PolicyError("duplicate final hard boundary")
    return {
        "final_actions": final,
        "suppression_evidence": evidence,
        "final_action_count": len(final),
        "hard_action_count": len(hard_positions),
    }


def full_fusion_replay(
    proposals: Sequence[dict[str, Any]],
    vad_actions: Sequence[dict[str, Any]],
    *,
    cluster_debounce_ms: int,
    cluster_boundary_radius_ms: int,
    refractory_ms: int,
    representative: str,
    detector_vad_radius_ms: int,
    same_silence_interval_association: bool,
    episode_observed_end: int,
    lifecycle_events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    clustered = cluster_proposals(
        proposals,
        cluster_debounce_ms=cluster_debounce_ms,
        cluster_boundary_radius_ms=cluster_boundary_radius_ms,
        refractory_ms=refractory_ms,
        representative=representative,
        episode_observed_end=episode_observed_end,
    )
    detector = actionize_clusters(clustered["clusters"])
    contextual_vad, contextual_detector, silence_intervals = derive_fusion_context(
        vad_actions,
        detector,
        lifecycle_events,
    )
    fused = fuse_actions(
        contextual_vad,
        contextual_detector,
        detector_vad_radius_ms=detector_vad_radius_ms,
        same_silence_interval_association=same_silence_interval_association,
    )
    return {
        "cluster": clustered,
        "detector_actions": contextual_detector,
        "silence_intervals": silence_intervals,
        "fusion": fused,
    }


def trace_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json(row).encode("utf-8") + b"\n")
    return digest.hexdigest()


def policy_progress(
    proposal_progress: Sequence[dict[str, Any]],
    clusters: Sequence[dict[str, Any]],
    *,
    episode_observed_end: int,
) -> list[dict[str, int]]:
    observations = sorted(
        {int(row["observed_source_sample"]) for row in proposal_progress}
        | {int(row["cluster_open_frontier"]) for row in clusters}
        | {int(row["observed_source_sample_at_emit"]) for row in clusters}
        | {episode_observed_end}
    )
    rows: list[dict[str, int]] = []
    raw_safe = 0
    progress_index = 0
    ordered_progress = sorted(
        proposal_progress,
        key=lambda row: (
            int(row["observed_source_sample"]),
            int(row["safe_boundary_frontier_sample"]),
        ),
    )
    for observed in observations:
        while (
            progress_index < len(ordered_progress)
            and int(ordered_progress[progress_index]["observed_source_sample"]) <= observed
        ):
            raw_safe = int(ordered_progress[progress_index]["safe_boundary_frontier_sample"])
            progress_index += 1
        pending = [
            int(row["boundary_source_sample"]) - 1
            for row in clusters
            if int(row["cluster_open_frontier"])
            <= observed
            < int(row["observed_source_sample_at_emit"])
        ]
        safe = min([raw_safe, *pending]) if pending else raw_safe
        safe = max(0, min(observed, safe))
        if rows and safe < rows[-1]["safe_boundary_frontier_sample"]:
            raise Phase5PolicyError("policy safe frontier regressed")
        rows.append(
            {
                "observed_source_sample": observed,
                "safe_boundary_frontier_sample": safe,
            }
        )
    if rows and rows[-1]["safe_boundary_frontier_sample"] != episode_observed_end:
        raise Phase5PolicyError("policy safe frontier did not close at episode end")
    return rows


def detector_created_hard_actions(
    final_actions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in final_actions
            if row.get("origin") == "detector"
            and row.get("action_kind") in ("add_hard_boundary", "accelerate_or_replace_vad")
        ],
        key=action_order,
    )


def validate_active_intervals(intervals: Sequence[dict[str, Any]]) -> None:
    previous_end = -1
    for row in sorted(intervals, key=lambda item: (int(item["start"]), int(item["end"]))):
        start = int(row["start"])
        end = int(row["end"])
        start_observed = int(row["start_observed_source_sample"])
        end_observed = int(row["end_observed_source_sample"])
        if start < previous_end or end <= start:
            raise Phase5PolicyError("VAD-active intervals overlap or are empty")
        if start_observed < start or end_observed < end or end_observed < start_observed:
            raise Phase5PolicyError("VAD-active interval has anticipatory evidence")
        previous_end = end


def causal_active_points(
    intervals: Sequence[dict[str, Any]], availability: int, quantum: int = 512
) -> list[int]:
    points: list[int] = []
    for row in intervals:
        if int(row["start_observed_source_sample"]) > availability:
            continue
        start = int(row["start"])
        if int(row["end_observed_source_sample"]) <= availability:
            end = int(row["end"])
        else:
            end = min(int(row["end"]), availability)
        first = ((start + quantum - 1) // quantum) * quantum
        points.extend(range(first, end, quantum))
    return sorted(set(point for point in points if point <= availability))


def choose_nearest_unused(
    points: Sequence[int], target: int, used: set[int], forbidden: set[int]
) -> int | None:
    eligible = [point for point in points if point not in used and point not in forbidden]
    if not eligible:
        return None
    return min(eligible, key=lambda point: (abs(point - target), point))


class Fenwick:
    def __init__(self, size: int) -> None:
        self.values = [0] * (size + 1)

    def add(self, index: int, delta: int) -> None:
        index += 1
        while index < len(self.values):
            self.values[index] += delta
            index += index & -index

    def prefix(self, end: int) -> int:
        total = 0
        while end > 0:
            total += self.values[end]
            end -= end & -end
        return total

    def total(self) -> int:
        return self.prefix(len(self.values) - 1)

    def kth(self, order: int) -> int:
        if order <= 0 or order > self.total():
            raise Phase5PolicyError("ordered causal point is unavailable")
        index = 0
        bit = 1 << ((len(self.values) - 1).bit_length() - 1)
        while bit:
            candidate = index + bit
            if candidate < len(self.values) and self.values[candidate] < order:
                index = candidate
                order -= self.values[candidate]
            bit >>= 1
        return index


def causal_active_point_schedule(
    intervals: Sequence[dict[str, Any]], quantum: int = 512
) -> tuple[list[int], list[int]]:
    schedule: dict[int, int] = {}
    for row in intervals:
        start = int(row["start"])
        end = int(row["end"])
        start_observed = int(row["start_observed_source_sample"])
        end_observed = int(row["end_observed_source_sample"])
        first = ((start + quantum - 1) // quantum) * quantum
        for point in range(first, end, quantum):
            available = max(start_observed, point if end_observed <= point else point + 1)
            schedule[point] = min(schedule.get(point, available), available)
    points = sorted(schedule)
    return points, [schedule[point] for point in points]


def nearest_tree_point(points: Sequence[int], tree: Fenwick, target: int) -> int | None:
    if not tree.total():
        return None
    index = bisect_left(points, target)
    left_count = tree.prefix(min(index + 1, len(points)))
    left_index = tree.kth(left_count) if left_count else None
    right_before = tree.prefix(index)
    right_index = tree.kth(right_before + 1) if tree.total() > right_before else None
    candidates = [
        points[candidate] for candidate in (left_index, right_index) if candidate is not None
    ]
    return min(candidates, key=lambda point: (abs(point - target), point)) if candidates else None


def control_action(
    kind: str,
    ordinal: int,
    boundary: int,
    availability: int,
    seed_material: str,
) -> dict[str, Any]:
    identity = {
        "control_kind": kind,
        "ordinal": ordinal,
        "boundary_source_sample": boundary,
        "observed_source_sample_at_emit": availability,
        "seed_material": seed_material,
    }
    return {
        "action_id": "control:" + content_sha256(identity)[:32],
        "origin": "control",
        "owner": "detector",
        "action_kind": "add_hard_boundary",
        "control_kind": kind,
        "boundary_source_sample": boundary,
        "observed_source_sample_at_emit": availability,
    }


def frequency_matched_control_reference(
    kind: str,
    neural_final_actions: Sequence[dict[str, Any]],
    active_intervals: Sequence[dict[str, Any]],
    *,
    energy_candidates: Sequence[dict[str, Any]],
    forbidden_boundaries: Sequence[int],
    seed_material: str,
) -> dict[str, Any]:
    if kind not in (
        "uniform_vad_active",
        "causal_energy_change_peak",
        "within_vad_active_position_shuffle",
    ):
        raise Phase5PolicyError(f"unsupported control kind: {kind}")
    validate_active_intervals(active_intervals)
    neural = detector_created_hard_actions(neural_final_actions)
    used: set[int] = set()
    forbidden = {int(value) for value in forbidden_boundaries}
    actions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    shuffle_candidates: list[dict[str, int]] = []
    for source_action in neural:
        source_availability = int(source_action["observed_source_sample_at_emit"])
        source_points = causal_active_points(active_intervals, source_availability)
        projected = choose_nearest_unused(
            source_points,
            int(source_action["boundary_source_sample"]),
            set(),
            forbidden,
        )
        if projected is not None:
            shuffle_candidates.append(
                {
                    "boundary_source_sample": projected,
                    "source_availability": source_availability,
                }
            )
    for ordinal, source_action in enumerate(neural):
        availability = int(source_action["observed_source_sample_at_emit"])
        points = causal_active_points(active_intervals, availability)
        selected: int | None = None
        if kind == "uniform_vad_active" and points:
            rank = ((ordinal + 1) * len(points)) // (len(neural) + 1)
            target = points[min(rank, len(points) - 1)]
            selected = choose_nearest_unused(points, target, used, forbidden)
        elif kind == "causal_energy_change_peak":
            eligible = [
                row
                for row in energy_candidates
                if int(row["observed_source_sample"]) <= availability
                and int(row["boundary_source_sample"]) in points
                and int(row["boundary_source_sample"]) not in used
                and int(row["boundary_source_sample"]) not in forbidden
            ]
            if eligible:
                selected = int(
                    min(
                        eligible,
                        key=lambda row: (
                            -float(row["change_strength"]),
                            int(row["observed_source_sample"]),
                            int(row["boundary_source_sample"]),
                            str(row["candidate_id"]),
                        ),
                    )["boundary_source_sample"]
                )
        elif kind == "within_vad_active_position_shuffle" and points:
            eligible = [
                row
                for row in shuffle_candidates
                if int(row["boundary_source_sample"]) in points
                and int(row["boundary_source_sample"]) not in used
                and int(row["boundary_source_sample"]) not in forbidden
            ]
            if eligible:
                chosen = min(
                    eligible,
                    key=lambda row: (
                        hashlib.sha256(
                            (
                                f"{seed_material}|{ordinal}|{availability}|"
                                f"{row['boundary_source_sample']}|{row['source_availability']}"
                            ).encode("utf-8")
                        ).hexdigest(),
                        int(row["boundary_source_sample"]),
                        int(row["source_availability"]),
                    ),
                )
                selected = int(chosen["boundary_source_sample"])
        if selected is None:
            failures.append(
                {
                    "ordinal": ordinal,
                    "source_action_id": str(source_action["final_action_id"]),
                    "availability": availability,
                    "reason": "no_distinct_causal_vad_active_placement",
                }
            )
            continue
        used.add(selected)
        created = control_action(kind, ordinal, selected, availability, seed_material)
        created["final_action_id"] = f"final:{created['action_id']}"
        created["source_session_id"] = str(source_action["source_session_id"])
        created["audio_epoch"] = int(source_action["audio_epoch"])
        actions.append(created)
    complete = not failures
    return {
        "control_kind": kind,
        "required_hard_action_count": len(neural),
        "placed_hard_action_count": len(actions),
        "status": "complete" if complete else "infeasible",
        "actions": actions,
        "infeasible_placements": failures,
        "causal_availability_exactly_preserved": complete
        and all(
            int(action["observed_source_sample_at_emit"])
            == int(neural[index]["observed_source_sample_at_emit"])
            for index, action in enumerate(actions)
        ),
        "ground_truth_inputs_used": False,
    }


def frequency_matched_control(
    kind: str,
    neural_final_actions: Sequence[dict[str, Any]],
    active_intervals: Sequence[dict[str, Any]],
    *,
    energy_candidates: Sequence[dict[str, Any]],
    forbidden_boundaries: Sequence[int],
    seed_material: str,
) -> dict[str, Any]:
    if kind not in (
        "uniform_vad_active",
        "causal_energy_change_peak",
        "within_vad_active_position_shuffle",
    ):
        raise Phase5PolicyError(f"unsupported control kind: {kind}")
    validate_active_intervals(active_intervals)
    neural = detector_created_hard_actions(neural_final_actions)
    used: set[int] = set()
    forbidden = {int(value) for value in forbidden_boundaries}
    points, point_availability = causal_active_point_schedule(active_intervals)
    point_index = {point: index for index, point in enumerate(points)}
    point_events = sorted((available, index) for index, available in enumerate(point_availability))
    actions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    shuffle_candidates: list[dict[str, Any]] = []
    projection_selectable = Fenwick(len(points))
    projection_event_index = 0
    for source_action in neural:
        source_availability = int(source_action["observed_source_sample_at_emit"])
        while (
            projection_event_index < len(point_events)
            and point_events[projection_event_index][0] <= source_availability
        ):
            _, index = point_events[projection_event_index]
            if points[index] not in forbidden:
                projection_selectable.add(index, 1)
            projection_event_index += 1
        projected = nearest_tree_point(
            points, projection_selectable, int(source_action["boundary_source_sample"])
        )
        if projected is not None:
            shuffle_candidates.append(
                {
                    "boundary_source_sample": projected,
                    "source_availability": source_availability,
                    "source_action_id": str(source_action["final_action_id"]),
                }
            )
    eligible_all = Fenwick(len(points))
    selectable = Fenwick(len(points))
    event_index = 0
    energy_events = sorted(
        (
            max(
                int(row["observed_source_sample"]),
                point_availability[point_index[int(row["boundary_source_sample"])]],
            ),
            str(row["candidate_id"]),
            row,
        )
        for row in energy_candidates
        if int(row["boundary_source_sample"]) in point_index
    )
    shuffle_events = sorted(
        (
            int(row["source_availability"]),
            str(row["source_action_id"]),
            row,
        )
        for row in shuffle_candidates
    )
    energy_event_index = 0
    shuffle_event_index = 0
    energy_heap: list[tuple[float, int, int, str, dict[str, Any]]] = []
    shuffle_heap: list[tuple[str, int, int, str, dict[str, Any]]] = []
    for ordinal, source_action in enumerate(neural):
        availability = int(source_action["observed_source_sample_at_emit"])
        while event_index < len(point_events) and point_events[event_index][0] <= availability:
            _, index = point_events[event_index]
            eligible_all.add(index, 1)
            if points[index] not in forbidden:
                selectable.add(index, 1)
            event_index += 1
        while (
            energy_event_index < len(energy_events)
            and energy_events[energy_event_index][0] <= availability
        ):
            _, _, row = energy_events[energy_event_index]
            heappush(
                energy_heap,
                (
                    -float(row["change_strength"]),
                    int(row["observed_source_sample"]),
                    int(row["boundary_source_sample"]),
                    str(row["candidate_id"]),
                    row,
                ),
            )
            energy_event_index += 1
        while (
            shuffle_event_index < len(shuffle_events)
            and shuffle_events[shuffle_event_index][0] <= availability
        ):
            _, _, row = shuffle_events[shuffle_event_index]
            rank = hashlib.sha256(
                (
                    f"{seed_material}|{row['boundary_source_sample']}|"
                    f"{row['source_availability']}|{row['source_action_id']}"
                ).encode("utf-8")
            ).hexdigest()
            heappush(
                shuffle_heap,
                (
                    rank,
                    int(row["boundary_source_sample"]),
                    int(row["source_availability"]),
                    str(row["source_action_id"]),
                    row,
                ),
            )
            shuffle_event_index += 1
        selected: int | None = None
        if kind == "uniform_vad_active" and eligible_all.total():
            rank = ((ordinal + 1) * eligible_all.total()) // (len(neural) + 1)
            target = points[eligible_all.kth(min(rank + 1, eligible_all.total()))]
            selected = nearest_tree_point(points, selectable, target)
        elif kind == "causal_energy_change_peak":
            while energy_heap and (
                int(energy_heap[0][4]["boundary_source_sample"]) in used
                or int(energy_heap[0][4]["boundary_source_sample"]) in forbidden
            ):
                heappop(energy_heap)
            if energy_heap:
                selected = int(energy_heap[0][4]["boundary_source_sample"])
        elif kind == "within_vad_active_position_shuffle":
            while shuffle_heap and (
                int(shuffle_heap[0][4]["boundary_source_sample"]) in used
                or int(shuffle_heap[0][4]["boundary_source_sample"]) in forbidden
            ):
                heappop(shuffle_heap)
            if shuffle_heap:
                selected = int(shuffle_heap[0][4]["boundary_source_sample"])
        if selected is None:
            failures.append(
                {
                    "ordinal": ordinal,
                    "source_action_id": str(source_action["final_action_id"]),
                    "availability": availability,
                    "reason": "no_distinct_causal_vad_active_placement",
                }
            )
            continue
        used.add(selected)
        selectable.add(point_index[selected], -1)
        created = control_action(kind, ordinal, selected, availability, seed_material)
        created["final_action_id"] = f"final:{created['action_id']}"
        created["source_session_id"] = str(source_action["source_session_id"])
        created["audio_epoch"] = int(source_action["audio_epoch"])
        actions.append(created)
    complete = not failures
    return {
        "control_kind": kind,
        "required_hard_action_count": len(neural),
        "placed_hard_action_count": len(actions),
        "status": "complete" if complete else "infeasible",
        "actions": actions,
        "infeasible_placements": failures,
        "causal_availability_exactly_preserved": complete
        and all(
            int(action["observed_source_sample_at_emit"])
            == int(neural[index]["observed_source_sample_at_emit"])
            for index, action in enumerate(actions)
        ),
        "ground_truth_inputs_used": False,
    }
