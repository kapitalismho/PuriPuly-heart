from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .build_episodes import sha256_bytes, sha256_file
from .pcm_oracle import (
    CLAMP_SCHEMA_VERSION,
    DELAYS_MS,
    DETAIL_SCHEMA_VERSION,
    EXPECTED_ACTION_INSTANCES,
    EXPECTED_ACTIONS_PER_SHARD,
    EXPECTED_CLAMP_ABOVE,
    EXPECTED_CLAMP_BELOW,
    EXPECTED_CLAMPED_ACTION_INSTANCES,
    EXPECTED_CLAMPED_EPISODES,
    EXPECTED_CLAMPED_REFERENCE_OFFSETS,
    EXPECTED_CLAMPED_REFERENCES,
    EXPECTED_DETAIL_ROWS,
    EXPECTED_EPISODE_COUNT,
    EXPECTED_GRID_ROWS,
    EXPECTED_HARD_POSITIVE_EPISODES,
    EXPECTED_NO_HARD_EPISODES,
    EXPECTED_POPULATION_SHA256,
    EXPECTED_REFERENCE_COUNT,
    EXPECTED_ROWS_PER_SHARD,
    EXPECTED_SESSION_COUNT,
    GRID_ID,
    HOLDBACKS_MS,
    OFFSETS_MS,
    OWNER_THRESHOLDS_MS,
    SAMPLES_PER_MS,
    VERIFICATION_SCHEMA_VERSION,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _nearest(values: list[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    return ordered[max(1, math.ceil(quantile * len(ordered))) - 1]


def _cover(spans: list[dict[str, Any]], start: int, end: int) -> dict[str, bool]:
    if start == end and not spans:
        return {"conservation": True, "no_duplication": True, "ordering": True}
    ordered = sorted(spans, key=lambda span: (int(span["start"]), int(span["end"])))
    cursor = start
    total = 0
    disjoint = True
    ordering = True
    for span in ordered:
        left = int(span["start"])
        right = int(span["end"])
        if left != cursor or right <= left:
            ordering = False
        if left < cursor:
            disjoint = False
        cursor = max(cursor, right)
        total += max(0, right - left)
    conservation = bool(ordered) and int(ordered[0]["start"]) == start
    conservation = conservation and cursor == end and ordering and total == end - start
    return {
        "conservation": conservation,
        "no_duplication": disjoint and total == end - start,
        "ordering": ordering,
    }


def _ownership_valid(spans: list[dict[str, Any]]) -> bool:
    ordered = sorted(spans, key=lambda span: (int(span["start"]), int(span["end"])))
    return all(str(span["turn_id"]) == f"turn-{index:04d}" for index, span in enumerate(ordered))


def _boundary_turn_ids(spans: list[dict[str, Any]], boundary_sample: int) -> dict[str, str | None]:
    old_turn_id: str | None = None
    new_turn_id: str | None = None
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        if start < boundary_sample <= end:
            old_turn_id = str(span["turn_id"])
        if start <= boundary_sample < end:
            new_turn_id = str(span["turn_id"])
    return {"old_turn_id": old_turn_id, "new_turn_id": new_turn_id}


def _boundary_turn_ownership(spans: list[dict[str, Any]], boundary_sample: int) -> dict[str, Any]:
    ids = _boundary_turn_ids(spans, boundary_sample)
    old_span: list[int] | None = None
    new_span: list[int] | None = None
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        if str(span["turn_id"]) == ids["old_turn_id"]:
            old_span = [start, end]
        if str(span["turn_id"]) == ids["new_turn_id"]:
            new_span = [start, end]
    return {**ids, "old_turn_span": old_span, "new_turn_span": new_span}


def _contamination(
    spans: list[dict[str, Any]], singleton: list[list[Any]], threshold_ms: int
) -> dict[str, int]:
    threshold = threshold_ms * SAMPLES_PER_MS
    contaminated = 0
    denominator = sum(int(interval[1]) - int(interval[0]) for interval in singleton)
    for span in spans:
        left = int(span["start"])
        right = int(span["end"])
        qualifying: list[tuple[int, int, str]] = []
        for raw_start, raw_end, raw_speaker in singleton:
            start = max(left, int(raw_start))
            end = min(right, int(raw_end))
            if end - start >= threshold:
                qualifying.append((start, end, str(raw_speaker)))
        qualifying.sort(key=lambda item: (item[0], item[1], item[2]))
        if qualifying:
            owner = qualifying[0][2]
            contamination_started = False
            for start, end, speaker in qualifying:
                if speaker != owner:
                    contamination_started = True
                if contamination_started:
                    contaminated += end - start
    return {"contaminated_samples": contaminated, "denominator_samples": denominator}


def _progress_valid(row: dict[str, Any], field: str, digest_field: str) -> bool:
    progress = row[field]
    if _hash(progress) != row[digest_field]:
        return False
    previous_observed = int(row["epoch_origin_source_sample"])
    previous_safe = max(0, previous_observed - 1)
    for observed, safe in progress:
        observed = int(observed)
        safe = int(safe)
        if observed < previous_observed or safe < previous_safe or safe > observed:
            return False
        previous_observed = observed
        previous_safe = safe
    return True


def _exact_progress_valid(row: dict[str, Any], *, baseline: bool) -> bool:
    field = "baseline_progress_rows" if baseline else "progress_rows"
    progress = row[field]
    actions = [] if baseline else row["oracle_actions"]
    safe_by_observed: dict[int, int] = {}
    for observed_raw, safe_raw in progress:
        observed = int(observed_raw)
        safe = int(safe_raw)
        safe_by_observed[observed] = safe
        pending = [
            int(action["boundary_source_sample"])
            for action in actions
            if action["apply_frontier"] is None or int(action["apply_frontier"]) >= observed
        ]
        expected = max(0, min(observed, min(pending) - 1)) if pending else observed
        if safe != expected:
            return False
    for action in actions:
        apply_frontier = action["apply_frontier"]
        if apply_frontier is None:
            continue
        boundary = int(action["boundary_source_sample"])
        safe = safe_by_observed.get(int(apply_frontier))
        zero_sentinel = int(row["epoch_origin_source_sample"]) == 0 and boundary == 0
        if safe is None or (boundary <= safe and not zero_sentinel):
            return False
    return True


def _register_definitions(row: dict[str, Any], cache: dict[str, dict[str, Any]]) -> list[str]:
    mismatches: list[str] = []
    for definition in row.get("shared_definitions", []):
        payload = definition.get("payload")
        payload_hash = _hash(payload)
        definition_id = str(definition.get("definition_id"))
        kind = str(definition.get("kind"))
        if definition.get("payload_sha256") != payload_hash:
            mismatches.append("shared_definition_payload_hash")
        if definition_id != f"{kind}:{payload_hash}":
            mismatches.append("shared_definition_id")
        existing = cache.get(definition_id)
        if existing is not None and existing != payload:
            mismatches.append("shared_definition_collision")
        cache[definition_id] = payload
    return mismatches


def _hydrate_row(
    row: dict[str, Any], cache: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any] | None, list[str]]:
    mismatches: list[str] = []
    refs = {
        "episode_static": str(row.get("episode_static_ref")),
        "baseline_evidence": str(row.get("baseline_evidence_ref")),
        "candidate_progress": str(row.get("candidate_progress_ref")),
    }
    payloads: dict[str, dict[str, Any]] = {}
    for kind, definition_id in refs.items():
        payload = cache.get(definition_id)
        if payload is None:
            mismatches.append(f"missing_shared_definition:{kind}")
        elif not definition_id.startswith(f"{kind}:"):
            mismatches.append(f"shared_definition_kind:{kind}")
        else:
            payloads[kind] = payload
    if mismatches:
        return None, mismatches
    expanded = dict(row)
    static = payloads["episode_static"]
    baseline = payloads["baseline_evidence"]
    progress = payloads["candidate_progress"]
    expanded["singleton_intervals"] = static["singleton_intervals"]
    expanded["singleton_intervals_sha256"] = static["singleton_intervals_sha256"]
    expanded["lifecycle_events"] = static["lifecycle_events"]
    expanded["b0_actions"] = baseline["b0_actions"]
    expanded["baseline_progress_rows"] = baseline["baseline_progress_rows"]
    expanded["baseline_progress_sha256"] = baseline["baseline_progress_sha256"]
    expanded["baseline_turn_spans"] = baseline["baseline_turn_spans"]
    expanded["baseline_drain_records"] = baseline["baseline_drain_records"]
    expanded["baseline_finalization_records"] = baseline["baseline_finalization_records"]
    expanded["baseline_state_digest"] = baseline["baseline_state_digest"]
    expanded["progress_rows"] = progress["progress_rows"]
    expanded["progress_sha256"] = progress["progress_sha256"]
    expanded["metrics"] = dict(row["metrics"])
    expanded["metrics"]["baseline"] = baseline["baseline_metrics"]
    expanded["invariants"] = dict(row["invariants"])
    expanded["invariants"]["baseline"] = baseline["baseline_invariants"]
    return expanded, mismatches


@dataclass(slots=True)
class IndependentAccumulator:
    delay_ms: int
    offset_ms: int
    holdback_ms: int
    detail_rows: int = 0
    action_instances: int = 0
    clamp_instances: int = 0
    recoverable: int = 0
    late: int = 0
    unavailable: int = 0
    unrecoverable: list[int] = field(default_factory=list)
    fragments: list[int] = field(default_factory=list)
    latencies: list[int] = field(default_factory=list)
    invariant_failures: int = 0
    safe_complete: int = 0
    fallback: int = 0
    improved: int = 0
    unchanged: int = 0
    regressed: int = 0
    metrics: dict[str, dict[str, dict[str, dict[str, int]]]] = field(
        default_factory=lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: {"contaminated_samples": 0, "denominator_samples": 0})
            )
        )
    )
    sessions: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(
            lambda: {"baseline": 0, "candidate": 0, "denominator": 0}
        )
    )

    def add(self, row: dict[str, Any]) -> None:
        self.detail_rows += 1
        actions = row["oracle_actions"]
        self.action_instances += len(actions)
        self.clamp_instances += int(row["clamp_count"])
        for action in actions:
            span = action["unrecoverable_span"]
            self.unrecoverable.append(int(span[1]) - int(span[0]) if span else 0)
            label = action["recoverability"]
            if label == "fully_recoverable":
                self.recoverable += 1
            elif label == "late_unrecoverable":
                self.late += 1
            elif label == "unavailable_before_terminal":
                self.unavailable += 1
        self.fragments.extend(int(value) for value in row["fragment_durations_samples"])
        self.latencies.extend(int(value) for value in row["logical_action_latencies_samples"])
        for system in ("baseline", "candidate"):
            if not all(bool(value) for value in row["invariants"][system].values()):
                self.invariant_failures += 1
        for drain in row["candidate_drain_records"]:
            if drain["outcome"] == "safe_complete":
                self.safe_complete += 1
            elif drain["outcome"] == "safe_drain_timeout_fallback":
                self.fallback += 1
        tag = str(row["tag"])
        for system in ("baseline", "candidate"):
            for threshold, values in row["metrics"][system].items():
                target = self.metrics[tag][system][threshold]
                target["contaminated_samples"] += int(values["contaminated_samples"])
                target["denominator_samples"] += int(values["denominator_samples"])
        if tag == "hard_only":
            baseline = int(row["metrics"]["baseline"]["100"]["contaminated_samples"])
            candidate = int(row["metrics"]["candidate"]["100"]["contaminated_samples"])
            denominator = int(row["metrics"]["candidate"]["100"]["denominator_samples"])
            if candidate < baseline:
                self.improved += 1
            elif candidate == baseline:
                self.unchanged += 1
            else:
                self.regressed += 1
            session = self.sessions[str(row["session_id"])]
            session["baseline"] += baseline
            session["candidate"] += candidate
            session["denominator"] += denominator

    def finish(self) -> dict[str, Any]:
        metrics = json.loads(canonical_json(self.metrics))
        hard = metrics.get("hard_only", {})
        baseline = hard.get("baseline", {}).get(
            "100", {"contaminated_samples": 0, "denominator_samples": 0}
        )
        candidate = hard.get("candidate", {}).get(
            "100", {"contaminated_samples": 0, "denominator_samples": 0}
        )
        denominator = int(candidate["denominator_samples"])
        baseline_ratio = (
            int(baseline["contaminated_samples"]) / denominator if denominator else None
        )
        candidate_ratio = (
            int(candidate["contaminated_samples"]) / denominator if denominator else None
        )
        session_effects = []
        for session_id in sorted(self.sessions):
            values = self.sessions[session_id]
            session_denominator = values["denominator"]
            session_effects.append(
                {
                    "session_id": session_id,
                    "baseline_contaminated_samples": values["baseline"],
                    "candidate_contaminated_samples": values["candidate"],
                    "denominator_samples": session_denominator,
                    "paired_ratio_difference": (
                        (values["candidate"] - values["baseline"]) / session_denominator
                        if session_denominator
                        else None
                    ),
                }
            )
        total_unrecoverable = sum(self.unrecoverable)
        result = {
            "grid_row_id": f"d{self.delay_ms}:o{self.offset_ms}:h{self.holdback_ms}",
            "availability_delay_ms": self.delay_ms,
            "boundary_offset_ms": self.offset_ms,
            "holdback_ms": self.holdback_ms,
            "detail_rows": self.detail_rows,
            "action_instances": self.action_instances,
            "clamp_instances": self.clamp_instances,
            "invariant_failures": self.invariant_failures,
            "fully_recoverable_actions": self.recoverable,
            "late_unrecoverable_actions": self.late,
            "unavailable_before_terminal_actions": self.unavailable,
            "fully_recoverable_action_fraction": (
                self.recoverable / self.action_instances if self.action_instances else None
            ),
            "unrecoverable_samples": {
                "count": len(self.unrecoverable),
                "total": total_unrecoverable,
                "mean_ms": (
                    total_unrecoverable / len(self.unrecoverable) / SAMPLES_PER_MS
                    if self.unrecoverable
                    else None
                ),
                "p50": _nearest(self.unrecoverable, 0.50),
                "p95": _nearest(self.unrecoverable, 0.95),
                "max": max(self.unrecoverable, default=None),
            },
            "fragment_duration_samples": {
                "count": len(self.fragments),
                "p10": _nearest(self.fragments, 0.10),
                "p50": _nearest(self.fragments, 0.50),
                "p90": _nearest(self.fragments, 0.90),
            },
            "logical_action_latency_samples": {
                "count": len(self.latencies),
                "p50": _nearest(self.latencies, 0.50),
                "p95": _nearest(self.latencies, 0.95),
                "max": max(self.latencies, default=None),
            },
            "safe_drains": {
                "safe_complete": self.safe_complete,
                "safe_drain_timeout_fallback": self.fallback,
            },
            "episode_effects": {
                "improved": self.improved,
                "unchanged": self.unchanged,
                "regressed": self.regressed,
            },
            "contamination": metrics,
            "primary_hard_only_100ms": {
                "baseline_contamination_ratio": baseline_ratio,
                "candidate_contamination_ratio": candidate_ratio,
                "paired_ratio_difference": (
                    candidate_ratio - baseline_ratio
                    if candidate_ratio is not None and baseline_ratio is not None
                    else None
                ),
                "oracle_reduces_contamination": int(candidate["contaminated_samples"])
                < int(baseline["contaminated_samples"]),
            },
            "session_effects": session_effects,
        }
        result["grid_row_digest"] = _hash(result)
        return result


def _row_mismatches(row: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    if row.get("schema_version") != DETAIL_SCHEMA_VERSION or row.get("grid_id") != GRID_ID:
        mismatches.append("detail_schema")
    start = int(row["scored_start"])
    end = int(row["processed_scored_end"])
    for spans_field, invariant_key in (
        ("baseline_turn_spans", "baseline"),
        ("realized_turn_spans", "candidate"),
        ("ideal_turn_spans", "ideal"),
    ):
        flags = _cover(row[spans_field], start, end)
        if flags != row["invariants"][invariant_key]:
            mismatches.append(f"{spans_field}_flags")
        if not all(flags.values()) or not _ownership_valid(row[spans_field]):
            mismatches.append(f"{spans_field}_ownership")
    singleton = row["singleton_intervals"]
    if _hash(singleton) != row["singleton_intervals_sha256"]:
        mismatches.append("singleton_digest")
    for system, spans_field in (
        ("baseline", "baseline_turn_spans"),
        ("candidate", "realized_turn_spans"),
    ):
        for threshold in OWNER_THRESHOLDS_MS:
            actual = _contamination(row[spans_field], singleton, threshold)
            if actual != row["metrics"][system][str(threshold)]:
                mismatches.append(f"contamination_{system}_{threshold}")
    if not _progress_valid(row, "progress_rows", "progress_sha256"):
        mismatches.append("progress")
    elif not _exact_progress_valid(row, baseline=False):
        mismatches.append("progress_exact_safe_frontier")
    if not _progress_valid(row, "baseline_progress_rows", "baseline_progress_sha256"):
        mismatches.append("baseline_progress")
    elif not _exact_progress_valid(row, baseline=True):
        mismatches.append("baseline_progress_exact_safe_frontier")
    if row["final_ring_span"] != [end, end]:
        mismatches.append("terminal_ring")
    if row["terminal_release_record"]["released_through"] != end:
        mismatches.append("terminal_release")
    prior_target: int | None = None
    finalizations_by_id = {
        str(record["finalization_id"]): record for record in row["candidate_finalization_records"]
    }
    realized_boundaries = {int(span["start"]) for span in row["realized_turn_spans"][1:]}
    progress_by_observed = {int(observed): int(safe) for observed, safe in row["progress_rows"]}
    for drain in row["candidate_drain_records"]:
        target = int(drain["target_sample"])
        if prior_target is not None and target < prior_target:
            mismatches.append("drain_fifo_target_order")
        prior_target = target
        if drain["outcome"] not in ("safe_complete", "safe_drain_timeout_fallback"):
            mismatches.append("drain_outcome")
        if (
            drain["outcome"] == "safe_complete"
            and progress_by_observed.get(int(drain["resolution_observed_frontier"]), -1) < target
        ):
            mismatches.append("drain_safe_coverage")
        if drain["outcome"] == "safe_drain_timeout_fallback" and int(
            drain["resolution_clock_ms"]
        ) < int(drain["deadline_clock_ms"]):
            mismatches.append("early_fallback")
        finalization_record_id = drain.get("finalization_record_id")
        if finalization_record_id is None:
            mismatches.append("drain_missing_finalization")
        else:
            finalization = finalizations_by_id.get(str(finalization_record_id))
            if finalization is None:
                mismatches.append("drain_finalization_reference")
            else:
                if int(finalization["finalization_source_sample"]) != target:
                    mismatches.append("drain_finalization_boundary")
                creates_hard_boundary = finalization["event_reason"] == "max_duration"
                if finalization.get("creates_hard_boundary") != creates_hard_boundary:
                    mismatches.append("drain_finalization_taxonomy")
                if creates_hard_boundary and target not in realized_boundaries:
                    mismatches.append("max_duration_boundary_missing")
                if creates_hard_boundary and not finalization["hard_boundary_action_ids"]:
                    mismatches.append("max_duration_action_reference")
        if int(drain["released_frontier_after_resolution"]) < int(
            drain["released_frontier_before_resolution"]
        ):
            mismatches.append("drain_release_regression")
    lifecycle_speech_end_ids = {
        str(event["event_id"])
        for event in row["lifecycle_events"]
        if event["event_kind"] == "speech_end"
    }
    finalized_event_ids = {
        str(record["event_id"]) for record in row["candidate_finalization_records"]
    }
    if lifecycle_speech_end_ids != finalized_event_ids:
        mismatches.append("lifecycle_finalization_completeness")
    for action in row["oracle_actions"]:
        ownership = action.get("ownership")
        if not isinstance(ownership, dict):
            mismatches.append("oracle_ownership_missing")
            continue
        boundary = int(action["boundary_source_sample"])
        ideal = _boundary_turn_ownership(row["ideal_turn_spans"], boundary)
        realized_boundary = action["realized_boundary_source_sample"]
        realized = (
            _boundary_turn_ownership(row["realized_turn_spans"], int(realized_boundary))
            if realized_boundary is not None
            else {
                "old_turn_id": None,
                "new_turn_id": None,
                "old_turn_span": None,
                "new_turn_span": None,
            }
        )
        recorded_ideal = {
            "old_turn_id": ownership.get("ideal_old_turn_id"),
            "new_turn_id": ownership.get("ideal_new_turn_id"),
        }
        recorded_realized = {
            "old_turn_id": ownership.get("realized_old_turn_id"),
            "new_turn_id": ownership.get("realized_new_turn_id"),
        }
        recorded_ideal_spans = {
            "old_turn_span": ownership.get("ideal_old_turn_span"),
            "new_turn_span": ownership.get("ideal_new_turn_span"),
        }
        recorded_realized_spans = {
            "old_turn_span": ownership.get("realized_old_turn_span"),
            "new_turn_span": ownership.get("realized_new_turn_span"),
        }
        if recorded_ideal != {
            "old_turn_id": ideal["old_turn_id"],
            "new_turn_id": ideal["new_turn_id"],
        }:
            mismatches.append("oracle_ideal_ownership")
        if recorded_realized != {
            "old_turn_id": realized["old_turn_id"],
            "new_turn_id": realized["new_turn_id"],
        }:
            mismatches.append("oracle_realized_ownership")
        if recorded_ideal_spans != {
            "old_turn_span": ideal["old_turn_span"],
            "new_turn_span": ideal["new_turn_span"],
        }:
            mismatches.append("oracle_ideal_ownership_spans")
        if recorded_realized_spans != {
            "old_turn_span": realized["old_turn_span"],
            "new_turn_span": realized["new_turn_span"],
        }:
            mismatches.append("oracle_realized_ownership_spans")
        old_side_matches = (
            ideal["old_turn_span"] is None and realized["old_turn_span"] is None
        ) or (
            ideal["old_turn_span"] is not None
            and realized["old_turn_span"] is not None
            and ideal["old_turn_span"][1] == boundary
            and realized["old_turn_span"][1] == boundary
        )
        new_side_matches = (
            ideal["new_turn_span"] is None and realized["new_turn_span"] is None
        ) or (
            ideal["new_turn_span"] is not None
            and realized["new_turn_span"] is not None
            and ideal["new_turn_span"][0] == boundary
            and realized["new_turn_span"][0] == boundary
        )
        boundary_assignment_matches = (
            realized_boundary == boundary and old_side_matches and new_side_matches
        )
        if ownership.get("boundary_assignment_matches_ideal") != boundary_assignment_matches:
            mismatches.append("oracle_boundary_assignment_match_flag")
        expected_match = (
            action["recoverability"] != "fully_recoverable" or boundary_assignment_matches
        )
        if ownership.get("fully_recoverable_matches_ideal") != expected_match:
            mismatches.append("oracle_ownership_match_flag")
        if action["recoverability"] == "fully_recoverable" and not expected_match:
            mismatches.append("oracle_fully_recoverable_ownership")
    return mismatches


MUTATION_IDS = (
    "missing_row",
    "duplicated_span",
    "altered_ownership",
    "altered_contamination_numerator",
    "altered_quantile",
)


def _mutation_fixtures(main_path: Path) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for mutation_id in MUTATION_IDS:
        result = verify_artifact(main_path, mutation=mutation_id, run_mutations=False)
        fixtures.append(
            {
                "fixture_id": mutation_id,
                "rejected": not result["passed"],
                "rejection_mismatches": result["mismatches"],
            }
        )
    return fixtures


def verify_artifact(
    main_path: Path, *, mutation: str | None = None, run_mutations: bool = True
) -> dict[str, Any]:
    if mutation is not None and mutation not in MUTATION_IDS:
        raise ValueError(f"unknown Phase 3 verifier mutation: {mutation}")
    main = json.loads(main_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    content_payload = {key: value for key, value in main.items() if key != "content_sha256"}
    if _hash(content_payload) != main.get("content_sha256"):
        mismatches.append("main_content_sha256")
    if main.get("grid_id") != GRID_ID:
        mismatches.append("grid_id")
    expected_grid = {
        "availability_delays_ms": list(DELAYS_MS),
        "boundary_offsets_ms": list(OFFSETS_MS),
        "holdbacks_ms": list(HOLDBACKS_MS),
    }
    for key, expected in expected_grid.items():
        if main.get("grid", {}).get(key) != expected:
            mismatches.append(f"grid_{key}")
    accumulators = {
        (delay, offset, holdback): IndependentAccumulator(delay, offset, holdback)
        for delay in DELAYS_MS
        for offset in OFFSETS_MS
        for holdback in HOLDBACKS_MS
    }
    session_ids: set[str] = set()
    episode_ids: set[str] = set()
    reference_ids: set[str] = set()
    hard_positive_episode_ids: set[str] = set()
    identities: set[tuple[Any, ...]] = set()
    clamp_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    total_rows = 0
    total_actions = 0
    total_clamps = 0
    sample_row: dict[str, Any] | None = None
    mutation_applied = False
    shard_results: list[dict[str, Any]] = []
    for shard in main["shards"]:
        shard_path = main_path.parent / str(shard["path"])
        byte_hash = sha256_file(shard_path)
        if byte_hash != shard["byte_sha256"]:
            mismatches.append(f"shard_byte_hash:{shard_path.name}")
        identity_hasher = hashlib.sha256()
        shard_rows = 0
        shard_actions = 0
        shard_clamps = 0
        previous_order: tuple[Any, ...] | None = None
        definition_cache: dict[str, dict[str, Any]] = {}
        with gzip.open(shard_path, "rt", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                compacted = json.loads(line)
                recorded_digest = compacted.get("row_digest")
                digest_payload = {
                    key: value for key, value in compacted.items() if key != "row_digest"
                }
                if _hash(digest_payload) != recorded_digest:
                    mismatches.append(f"{shard_path.name}:{line_number}:row_digest")
                definition_errors = _register_definitions(compacted, definition_cache)
                mismatches.extend(
                    f"{shard_path.name}:{line_number}:{error}" for error in definition_errors
                )
                row, hydration_errors = _hydrate_row(compacted, definition_cache)
                mismatches.extend(
                    f"{shard_path.name}:{line_number}:{error}" for error in hydration_errors
                )
                if row is None:
                    continue
                if mutation == "missing_row" and not mutation_applied:
                    mutation_applied = True
                    continue
                if mutation == "duplicated_span" and not mutation_applied:
                    row = json.loads(canonical_json(row))
                    row["realized_turn_spans"].append(dict(row["realized_turn_spans"][0]))
                    mutation_applied = True
                elif mutation == "altered_ownership" and not mutation_applied:
                    if row["oracle_actions"]:
                        row = json.loads(canonical_json(row))
                        row["oracle_actions"][0]["ownership"]["realized_old_turn_id"] = "altered"
                        mutation_applied = True
                elif mutation == "altered_contamination_numerator" and not mutation_applied:
                    row = json.loads(canonical_json(row))
                    row["metrics"]["candidate"]["100"]["contaminated_samples"] += 1
                    mutation_applied = True
                if sample_row is None:
                    sample_row = row
                row_errors = _row_mismatches(row)
                mismatches.extend(
                    f"{shard_path.name}:{line_number}:{error}" for error in row_errors
                )
                identity = (
                    int(row["availability_delay_ms"]),
                    int(row["boundary_offset_ms"]),
                    int(row["holdback_ms"]),
                    str(row["session_id"]),
                    str(row["episode_id"]),
                )
                if previous_order is not None and identity < previous_order:
                    mismatches.append(f"shard_order:{shard_path.name}:{line_number}")
                previous_order = identity
                if identity in identities:
                    mismatches.append(f"duplicate_identity:{identity}")
                identities.add(identity)
                identity_payload = {
                    "availability_delay_ms": identity[0],
                    "boundary_offset_ms": identity[1],
                    "holdback_ms": identity[2],
                    "session_id": identity[3],
                    "episode_id": identity[4],
                }
                identity_hasher.update((canonical_json(identity_payload) + "\n").encode("utf-8"))
                session_ids.add(identity[3])
                episode_ids.add(identity[4])
                if row["oracle_actions"]:
                    hard_positive_episode_ids.add(identity[4])
                for action in row["oracle_actions"]:
                    reference_id = str(action["reference_id"])
                    reference_ids.add(reference_id)
                    if action["clamp_direction"] is not None:
                        key = (identity[4], reference_id, int(action["requested_offset_ms"]))
                        clamp_rows[key] = {
                            "episode_id": identity[4],
                            "reference_id": reference_id,
                            "offset_ms": int(action["requested_offset_ms"]),
                            "unclamped_boundary_source_sample": int(action["unclamped_boundary"]),
                            "clamp_direction": str(action["clamp_direction"]),
                            "boundary_source_sample": int(action["boundary_source_sample"]),
                        }
                key = (identity[0], identity[1], identity[2])
                accumulators[key].add(row)
                shard_rows += 1
                shard_actions += len(row["oracle_actions"])
                shard_clamps += int(row["clamp_count"])
        if shard_rows != shard["row_count"] or shard_rows != EXPECTED_ROWS_PER_SHARD:
            mismatches.append(f"shard_row_count:{shard_path.name}")
        if shard_actions != shard["action_count"] or shard_actions != EXPECTED_ACTIONS_PER_SHARD:
            mismatches.append(f"shard_action_count:{shard_path.name}")
        if shard_clamps != shard["clamp_count"]:
            mismatches.append(f"shard_clamp_count:{shard_path.name}")
        if identity_hasher.hexdigest() != shard["identity_digest"]:
            mismatches.append(f"shard_identity_digest:{shard_path.name}")
        shard_results.append(
            {
                "path": str(shard["path"]),
                "byte_sha256": byte_hash,
                "row_count": shard_rows,
                "action_count": shard_actions,
                "clamp_count": shard_clamps,
                "identity_digest": identity_hasher.hexdigest(),
            }
        )
        total_rows += shard_rows
        total_actions += shard_actions
        total_clamps += shard_clamps
    population_identity = {
        "session_ids": sorted(session_ids),
        "episode_ids": sorted(episode_ids),
        "reference_ids": sorted(reference_ids),
    }
    population_sha = _hash(population_identity)
    population_counts = {
        "sessions": len(session_ids),
        "episodes": len(episode_ids),
        "hard_references": len(reference_ids),
        "hard_positive_episodes": len(hard_positive_episode_ids),
        "no_hard_control_episodes": len(episode_ids - hard_positive_episode_ids),
    }
    expected_population_counts = {
        "sessions": EXPECTED_SESSION_COUNT,
        "episodes": EXPECTED_EPISODE_COUNT,
        "hard_references": EXPECTED_REFERENCE_COUNT,
        "hard_positive_episodes": EXPECTED_HARD_POSITIVE_EPISODES,
        "no_hard_control_episodes": EXPECTED_NO_HARD_EPISODES,
    }
    if population_sha != EXPECTED_POPULATION_SHA256:
        mismatches.append("population_sha256")
    if population_counts != expected_population_counts:
        mismatches.append("population_counts")
    if main["population"]["identity"] != population_identity:
        mismatches.append("main_population_identity")
    clamp_list = sorted(
        clamp_rows.values(),
        key=lambda row: (row["episode_id"], row["reference_id"], row["offset_ms"]),
    )
    clamp_identity = {
        "schema_version": CLAMP_SCHEMA_VERSION,
        "clamped_reference_offsets": clamp_list,
    }
    clamp_sha = _hash(clamp_identity)
    clamp_counts = {
        "reference_offsets": len(clamp_list),
        "references": len({row["reference_id"] for row in clamp_list}),
        "episodes": len({row["episode_id"] for row in clamp_list}),
        "below_start": sum(row["clamp_direction"] == "below_start" for row in clamp_list),
        "above_end": sum(row["clamp_direction"] == "above_end" for row in clamp_list),
        "action_instances": total_clamps,
    }
    expected_clamp_counts = {
        "reference_offsets": EXPECTED_CLAMPED_REFERENCE_OFFSETS,
        "references": EXPECTED_CLAMPED_REFERENCES,
        "episodes": EXPECTED_CLAMPED_EPISODES,
        "below_start": EXPECTED_CLAMP_BELOW,
        "above_end": EXPECTED_CLAMP_ABOVE,
        "action_instances": EXPECTED_CLAMPED_ACTION_INSTANCES,
    }
    if clamp_sha != main["clamp_identity"]["sha256"]:
        mismatches.append("clamp_sha256")
    if clamp_counts != expected_clamp_counts:
        mismatches.append("clamp_counts")
    recomputed_grid = [
        accumulators[(delay, offset, holdback)].finish()
        for delay in DELAYS_MS
        for offset in OFFSETS_MS
        for holdback in HOLDBACKS_MS
    ]
    recomputed_grid_sha = _hash(recomputed_grid)
    if mutation == "altered_quantile":
        main["grid_rows"][0]["unrecoverable_samples"]["p95"] = -1
        mutation_applied = True
    if recomputed_grid_sha != main["grid_aggregate_sha256"]:
        mismatches.append("grid_aggregate_sha256")
    if recomputed_grid != main["grid_rows"]:
        mismatches.append("grid_rows")
    completeness = {
        "grid_rows": len(recomputed_grid),
        "detail_rows": total_rows,
        "action_instances": total_actions,
        "clamped_action_instances": total_clamps,
        "rows_per_delay_shard": EXPECTED_ROWS_PER_SHARD,
        "actions_per_delay_shard": EXPECTED_ACTIONS_PER_SHARD,
    }
    expected_completeness = {
        "grid_rows": EXPECTED_GRID_ROWS,
        "detail_rows": EXPECTED_DETAIL_ROWS,
        "action_instances": EXPECTED_ACTION_INSTANCES,
        "clamped_action_instances": EXPECTED_CLAMPED_ACTION_INSTANCES,
        "rows_per_delay_shard": EXPECTED_ROWS_PER_SHARD,
        "actions_per_delay_shard": EXPECTED_ACTIONS_PER_SHARD,
    }
    if completeness != expected_completeness or completeness != main["completeness"]:
        mismatches.append("completeness")
    if len(identities) != EXPECTED_DETAIL_ROWS:
        mismatches.append("identity_count")
    if mutation is not None and not mutation_applied:
        mismatches.append(f"mutation_not_applied:{mutation}")
    if sample_row is None:
        mismatches.append("no_detail_rows")
        mutation_fixtures: list[dict[str, Any]] = []
    elif run_mutations:
        mutation_fixtures = _mutation_fixtures(main_path)
        if not all(fixture["rejected"] for fixture in mutation_fixtures):
            mismatches.append("mutation_fixtures")
    else:
        mutation_fixtures = []
    pcm_path = Path(__file__).resolve().parent / "pcm_oracle.py"
    live_verifier_hash = sha256_file(Path(__file__).resolve())
    live_pcm_hash = sha256_file(pcm_path)
    if main["generated_from"]["pcm_oracle.py"] != live_pcm_hash:
        mismatches.append("live_pcm_code_hash")
    if main["generated_from"]["verify_pcm_oracle.py"] != live_verifier_hash:
        mismatches.append("live_verifier_code_hash")
    result: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "artifact_role": "oracle_provider_neutral_verification",
        "generated_from": {
            "verify_pcm_oracle.py": live_verifier_hash,
            "pcm_oracle.py": live_pcm_hash,
        },
        "main_artifact": {
            "path": main_path.name,
            "byte_sha256": sha256_file(main_path),
            "content_sha256": main.get("content_sha256"),
        },
        "shards": shard_results,
        "population_sha256": population_sha,
        "population_counts": population_counts,
        "clamp_sha256": clamp_sha,
        "clamp_counts": clamp_counts,
        "completeness": completeness,
        "recomputed_grid_aggregate_sha256": recomputed_grid_sha,
        "mutation_fixtures": mutation_fixtures,
        "injected_mutation": mutation,
        "mismatches": mismatches,
        "passed": not mismatches,
    }
    result["content_sha256"] = _hash(result)
    return result
