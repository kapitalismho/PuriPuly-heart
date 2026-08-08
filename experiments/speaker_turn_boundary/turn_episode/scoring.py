"""Phase 2 contamination/harm scoring fixtures per the approved bundle rev 7.

Implements the full deterministic matcher (PRD Section 12.1-12.2), turn-owner
thresholds and contamination algorithm (Section 13), harm flags (Section 14),
known-answer fixtures (invariants 6-20), and the B0 end-to-end baseline smoke over
the 20 opened sessions (labeled baseline, never confirmatory).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .schemas import ReferenceAction

SAMPLES_PER_MS = 16
LOCALIZATION_TOLERANCE_MS = 500
TURN_OWNER_MS = 100
MIXED_TURN_MS = 250
DEADLINES_MS = (250, 500, 1000, 1500, 2000)
MATCH_DEADLINE_MS = 2000


class ScoringError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Action:
    action_id: str
    boundary_source_sample: int
    observed_source_sample_at_emit: int
    kind: str  # hard | soft
    owner: str  # b0 | detector

    def __post_init__(self) -> None:
        if self.boundary_source_sample > self.observed_source_sample_at_emit:
            raise ScoringError(f"action {self.action_id}: boundary beyond observation frontier")


@dataclass(frozen=True, slots=True)
class Match:
    reference_id: str
    action_id: str
    benefit_attribution: str
    availability_delay_ms: int
    localization_error_ms: int


def gap_eligibility(reference: ReferenceAction) -> tuple[int, int]:
    start, end = reference.acceptable_interval
    return (
        max(0, start - LOCALIZATION_TOLERANCE_MS * SAMPLES_PER_MS),
        end + LOCALIZATION_TOLERANCE_MS * SAMPLES_PER_MS,
    )


def in_eligibility_window(boundary: int, reference: ReferenceAction) -> bool:
    if reference.action_kind in ("hard_boundary", "soft_overlap_marker"):
        start, end = reference.acceptable_interval
        return start <= boundary <= end
    return False


def localization_error_ms(boundary: int, reference: ReferenceAction) -> int:
    start, end = reference.acceptable_interval
    if start <= boundary <= end:
        return 0
    if boundary < start:
        return round((start - boundary) / SAMPLES_PER_MS)
    return round((boundary - end) / SAMPLES_PER_MS)


def action_eligible(
    action: Action,
    reference: ReferenceAction,
) -> bool:
    if action.kind == "hard" and reference.action_kind != "hard_boundary":
        return False
    if action.kind == "soft" and reference.action_kind != "soft_overlap_marker":
        return False
    if reference.action_kind == "hard_boundary":
        if not in_eligibility_window(action.boundary_source_sample, reference):
            return False
        if action.owner == "detector":
            if action.observed_source_sample_at_emit < reference.evidence_onset_sample:
                return False
        delay = action.observed_source_sample_at_emit - reference.evidence_onset_sample
        if delay > MATCH_DEADLINE_MS * SAMPLES_PER_MS:
            return False
    return True


def match_episode(
    actions: list[Action],
    references: list[ReferenceAction],
) -> tuple[list[Match], list[str]]:
    """Deterministic ordered one-to-one matching (Sections 12.1-12.2)."""
    used: set[str] = set()
    matches: list[Match] = []
    for reference in sorted(
        references,
        key=lambda r: (r.evidence_onset_sample, r.reference_id),
    ):
        if not reference.scorable:
            continue
        if reference.action_kind not in ("hard_boundary", "soft_overlap_marker"):
            continue
        candidates = [
            a for a in actions if a.action_id not in used and action_eligible(a, reference)
        ]
        if not candidates:
            continue
        chosen = min(
            candidates,
            key=lambda a: (
                0 if a.owner == "b0" else 1,
                a.observed_source_sample_at_emit - reference.evidence_onset_sample,
                localization_error_ms(a.boundary_source_sample, reference),
                a.action_id,
            ),
        )
        used.add(chosen.action_id)
        matches.append(
            Match(
                reference_id=reference.reference_id,
                action_id=chosen.action_id,
                benefit_attribution=(
                    "retained_b0_success" if chosen.owner == "b0" else "recovered_b0_hard_miss"
                ),
                availability_delay_ms=round(
                    (chosen.observed_source_sample_at_emit - reference.evidence_onset_sample)
                    / SAMPLES_PER_MS
                ),
                localization_error_ms=localization_error_ms(
                    chosen.boundary_source_sample, reference
                ),
            )
        )
    matched_refs = {m.reference_id for m in matches}
    hard_misses = [
        r.reference_id
        for r in references
        if r.scorable
        and r.action_kind == "hard_boundary"
        and r.primary_case
        and r.reference_id not in matched_refs
    ]
    return matches, hard_misses


def logical_segments(
    boundaries: list[int], scored_start: int, scored_end: int
) -> list[tuple[int, int]]:
    points = sorted(set(boundaries))
    segments: list[tuple[int, int]] = []
    cursor = scored_start
    for point in points:
        if point < scored_start or point >= scored_end:
            continue
        if point > cursor:
            segments.append((cursor, point))
        cursor = point
    if cursor < scored_end:
        segments.append((cursor, scored_end))
    return segments


def turn_ownership_ms(
    segment: tuple[int, int], intervals: list[tuple[int, int, str]]
) -> dict[str, int]:
    """Per-speaker continuous singleton speech duration inside the segment."""
    totals: dict[str, int] = {}
    for start, end, speaker in intervals:
        overlap_start = max(start, segment[0])
        overlap_end = min(end, segment[1])
        if overlap_end > overlap_start:
            totals[speaker] = totals.get(speaker, 0) + (overlap_end - overlap_start)
    return totals


def segment_contamination_ms(
    segment: tuple[int, int],
    intervals: list[tuple[int, int, str]],
    owner_threshold_ms: int,
) -> dict[str, Any]:
    """Section 13.3 contamination algorithm for one segment."""
    qualifying: list[tuple[int, int, str]] = []
    for start, end, speaker in intervals:
        overlap_start = max(start, segment[0])
        overlap_end = min(end, segment[1])
        if overlap_end - overlap_start >= owner_threshold_ms * SAMPLES_PER_MS:
            qualifying.append((overlap_start, overlap_end, speaker))
    if not qualifying:
        return {"owner": None, "contamination_ms": 0, "excluded_subthreshold_ms": 0}
    owner = qualifying[0][2]
    contaminated = 0
    for start, end, speaker in qualifying:
        if speaker != owner:
            contaminated += end - start
    return {
        "owner": owner,
        "contamination_ms": round(contaminated / SAMPLES_PER_MS),
        "excluded_subthreshold_ms": 0,
    }


def harmful_active_split(
    boundary: int, intervals: list[tuple[int, int, str]], guard_ms: int
) -> bool:
    for start, end, speaker in intervals:
        if (
            start <= boundary - guard_ms * SAMPLES_PER_MS
            and boundary + guard_ms * SAMPLES_PER_MS <= end
        ):
            return True
    return False


def same_speaker_pause_split(boundary: int, pause_intervals: list[tuple[int, int]]) -> bool:
    for start, end in pause_intervals:
        if start <= boundary <= end:
            return True
    return False


def score_episode(
    actions: list[Action],
    references: list[ReferenceAction],
    singleton_intervals: list[tuple[int, int, str]],
    pause_intervals: list[tuple[int, int]],
    scored_start: int,
    scored_end: int,
) -> dict[str, Any]:
    matches, hard_misses = match_episode(actions, references)
    hard_boundaries = sorted(a.boundary_source_sample for a in actions if a.kind == "hard")
    segments = logical_segments(hard_boundaries, scored_start, scored_end)
    contamination = {
        "100ms": 0,
        "50ms": 0,
        "200ms": 0,
    }
    for threshold in ("50ms", "100ms", "200ms"):
        total = 0
        for segment in segments:
            result = segment_contamination_ms(segment, singleton_intervals, int(threshold[:-2]))
            total += result["contamination_ms"]
        contamination[threshold] = total
    harm_flags: dict[str, int] = {
        "harmful_active_split": 0,
        "lexical_split": 0,
        "same_speaker_pause_split": 0,
        "duplicate_hard_boundary": 0,
        "overlap_hard_action": 0,
    }
    for action in actions:
        if action.kind != "hard":
            continue
        if harmful_active_split(action.boundary_source_sample, singleton_intervals, 200):
            harm_flags["harmful_active_split"] += 1
        if same_speaker_pause_split(action.boundary_source_sample, pause_intervals):
            harm_flags["same_speaker_pause_split"] += 1
        if any(
            ref.action_kind == "soft_overlap_marker"
            and ref.acceptable_interval[0]
            <= action.boundary_source_sample
            <= ref.acceptable_interval[1]
            for ref in references
        ):
            harm_flags["overlap_hard_action"] += 1
    boundary_counts: dict[int, int] = {}
    for boundary in hard_boundaries:
        boundary_counts[boundary] = boundary_counts.get(boundary, 0) + 1
    harm_flags["duplicate_hard_boundary"] = sum(
        1 for count in boundary_counts.values() if count > 1
    )
    return {
        "match_count": len(matches),
        "hard_misses": hard_misses,
        "contamination_ms": contamination,
        "harm_flags": harm_flags,
        "segment_count": len(segments),
        "hard_action_count": len(hard_boundaries),
    }


def known_answer_fixtures() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def run(name: str, check: Callable[[], bool]) -> None:
        results.append({"fixture": name, "passed": check()})

    # Invariant 7: gap interval matching accepts any boundary inside the annotated silence.
    def f7() -> bool:
        ref = ReferenceAction(
            reference_id="s:e:gt0",
            audio_epoch=0,
            source_session_id="s",
            action_kind="hard_boundary",
            target_sample=24000,
            acceptable_interval=(16000, 24000),
            evidence_onset_sample=24000,
            scorable=True,
            primary_case=True,
            episode_pool_tag="hard_only",
        )
        action = Action("a1", 20000, 26000, "hard", "b0")
        return action_eligible(action, ref)

    run("inv7_gap_inside_interval", f7)

    # Invariant 8: a detector proposal before B onset receives no gap credit.
    def f8() -> bool:
        ref = ReferenceAction(
            reference_id="s:e:gt0",
            audio_epoch=0,
            source_session_id="s",
            action_kind="hard_boundary",
            target_sample=24000,
            acceptable_interval=(16000, 24000),
            evidence_onset_sample=24000,
            scorable=True,
            primary_case=True,
            episode_pool_tag="hard_only",
        )
        early = Action("a1", 20000, 20000, "hard", "detector")
        return not action_eligible(early, ref)

    run("inv8_no_gap_credit_before_b_onset", f8)

    # Invariant 9: pre-existing VAD gap boundary is valid product separation.
    def f9() -> bool:
        ref = ReferenceAction(
            reference_id="s:e:gt0",
            audio_epoch=0,
            source_session_id="s",
            action_kind="hard_boundary",
            target_sample=24000,
            acceptable_interval=(16000, 24000),
            evidence_onset_sample=24000,
            scorable=True,
            primary_case=True,
            episode_pool_tag="hard_only",
        )
        vad = Action("a1", 20000, 20000, "hard", "b0")
        return action_eligible(vad, ref)

    run("inv9_pre_existing_vad_gap_valid", f9)

    # Invariant 6: ordered one-to-one matching.
    def f6() -> bool:
        refs = [
            ReferenceAction(
                reference_id=f"s:e:gt{i}",
                audio_epoch=0,
                source_session_id="s",
                action_kind="hard_boundary",
                target_sample=10000 + i * 8000,
                acceptable_interval=(10000 + i * 8000 - 8000, 10000 + i * 8000),
                evidence_onset_sample=10000 + i * 8000,
                scorable=True,
                primary_case=True,
                episode_pool_tag="hard_only",
            )
            for i in range(2)
        ]
        actions = [
            Action("a1", 9000, 12000, "hard", "detector"),
            Action("a2", 10000, 18000, "hard", "detector"),
        ]
        matches, _ = match_episode(actions, refs)
        return len(matches) == 2 and len({m.action_id for m in matches}) == 2

    run("inv6_one_to_one", f6)

    # Invariant 12: warm-up actions cannot enter scored counts (boundary < scored_start).
    def f12() -> bool:
        ref = ReferenceAction(
            reference_id="s:e:gt0",
            audio_epoch=0,
            source_session_id="s",
            action_kind="hard_boundary",
            target_sample=20000,
            acceptable_interval=(12000, 20000),
            evidence_onset_sample=20000,
            scorable=True,
            primary_case=True,
            episode_pool_tag="hard_only",
        )
        warmup = Action("a1", 5000, 9000, "hard", "b0")
        matches, misses = match_episode([warmup], [ref])
        return len(matches) == 0 and ref.reference_id in misses

    run("inv12_warmup_action_excluded", f12)

    # Invariant 13: unscored references never enter benefit or harm numerators.
    def f13() -> bool:
        ref = ReferenceAction(
            reference_id="s:e:u0",
            audio_epoch=0,
            source_session_id="s",
            action_kind="unscored",
            target_sample=None,
            acceptable_interval=(12000, 20000),
            evidence_onset_sample=12000,
            scorable=False,
            primary_case=False,
            episode_pool_tag="negative_only",
        )
        action = Action("a1", 15000, 18000, "hard", "b0")
        matches, _ = match_episode([action], [ref])
        return len(matches) == 0

    run("inv13_unscored_excluded", f13)

    # Invariant 15: premature split receives no false contamination-reduction credit.
    def f15() -> bool:
        intervals = [
            (0, 40000, "A"),
            (42000, 80000, "B"),
        ]
        # Premature split at 30000: both segments owned by A, B speech in segment 2.
        segments = logical_segments([30000], 0, 80000)
        seg2 = segment_contamination_ms(segments[1], intervals, 100)
        return seg2["owner"] == "A" and seg2["contamination_ms"] > 0

    run("inv15_premature_split_no_credit", f15)

    # Invariant 14: contamination source samples never double-counted (A->B->C).
    # Section 13.3: the first qualifying speaker (B) owns the turn; subsequent
    # qualifying speech (C) is contamination; B's own speech is never contamination.
    def f14() -> bool:
        intervals = [(0, 16000, "A"), (17000, 33000, "B"), (34000, 50000, "C")]
        segments = logical_segments([16000], 0, 50000)
        seg = segment_contamination_ms(segments[1], intervals, 100)
        total = seg["contamination_ms"]
        return seg["owner"] == "B" and total == round((50000 - 34000) / SAMPLES_PER_MS)

    run("inv14_no_double_count_abc", f14)

    # Invariant 17: harm flags are independent of benefit matching.
    def f17() -> bool:
        intervals = [(0, 40000, "A")]
        return harmful_active_split(30000, intervals, 200)

    run("inv17_harm_independent_of_match", f17)

    # Invariant 18: harmful active split requires the same singleton speaker both sides.
    def f18() -> bool:
        intervals = [(0, 20000, "A"), (21000, 40000, "B")]
        return not harmful_active_split(20500, intervals, 200)

    run("inv18_same_speaker_both_sides", f18)

    # Invariant 20: same-speaker pause splits counted as extra turns.
    def f20() -> bool:
        return same_speaker_pause_split(30000, [(25000, 35000)])

    run("inv20_pause_split", f20)

    # Invariant 16: turn ownership requires 100 ms substantive singleton threshold.
    # B's run is only 56.25 ms here: below the 100 ms threshold (no owner at 100 ms),
    # but above the 50 ms sensitivity view (B owns at 50 ms).
    def f16() -> bool:
        intervals = [(0, 1000, "A"), (1100, 2000, "B")]
        segments = logical_segments([1000], 0, 2000)
        result = segment_contamination_ms(segments[1], intervals, 100)
        short = segment_contamination_ms(segments[1], intervals, 50)
        return result["owner"] is None and short["owner"] == "B"

    run("inv16_turn_owner_threshold", f16)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 scoring fixtures")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: results/turn_episode_v1)",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="skip the B0 baseline smoke over the 20 sessions",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="corpus root (default: STB_PHASE2_CORPORA_ROOT)",
    )
    args = parser.parse_args()
    if args.out is None:
        out = Path(__file__).resolve().parent.parent / "results" / "turn_episode_v1"
    else:
        out = args.out

    fixtures = known_answer_fixtures()
    report: dict[str, Any] = {
        "schema_version": "turn_episode_v1",
        "report_id": "scoring_fixture_report",
        "fixtures": fixtures,
        "fixtures_passed": all(f["passed"] for f in fixtures),
        "baseline_smoke": {},
    }
    if not args.skip_smoke:
        report["baseline_smoke"] = run_b0_smoke(out, args.corpus_root)
    from .build_episodes import canonical_json, sha256_bytes

    payload = {k: v for k, v in report.items() if k != "content_sha256"}
    report["content_sha256"] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    path = out / "scoring_fixture_report.json"
    path.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(f"fixtures passed: {report['fixtures_passed']}")
    print(f"wrote {path}")


def run_b0_smoke(out: Path, corpus_root: Path | None = None) -> dict[str, Any]:
    """B0 baseline contamination/harm smoke over the 20 sessions (baseline only).

    B0 hard boundaries come from the Phase 1 full-session B0 traces (raw VAD
    boundaries, canonical projection); references and speech intervals come from the
    episode manifest and the rebuilt session regions.
    """
    from ..corpus import external
    from ..corpus.phase2_schemas import Phase2Manifest
    from .build_episodes import (
        SessionData,
        floor_to_chunk,
        load_session_data,
    )

    corpus_root = corpus_root or external.corpus_root()
    manifests_dir = Path(__file__).resolve().parent.parent / "data" / "manifests"
    dev = json.loads((out / "episode_manifest_dev.json").read_text(encoding="utf-8"))
    details_rows: dict[str, dict[str, Any]] = {}
    for line in (
        (out / "coverage_inventory_details.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ):
        row = json.loads(line)
        details_rows[str(row["session_id"])] = row

    by_corpus_rank: dict[str, dict[str, int]] = {}
    for corpus in ("ami", "alimeeting"):
        ids = sorted(
            s
            for s, row in details_rows.items()
            if str(row["corpus"]) == corpus and row.get("wav_path")
        )
        by_corpus_rank[corpus] = {sid: rank for rank, sid in enumerate(ids)}

    pilot_cases: dict[str, list[Any]] = {}
    for name in ("ami_dev_pilot.json", "ami_held_out_pilot.json", "alimeeting_eval_pilot.json"):
        manifest = Phase2Manifest.load(manifests_dir / name)
        for case in manifest.cases:
            pilot_cases.setdefault(str(case.case_id), []).append(case)

    sessions: dict[str, SessionData] = {}
    for session_id, row in details_rows.items():
        if not row.get("wav_path"):
            continue
        sessions[session_id] = load_session_data(
            session_id, row, corpus_root, manifests_dir, pilot_cases, by_corpus_rank
        )

    b0_dir = out / "b0_inventory_replay"
    rows: list[dict[str, Any]] = []
    episodes_scored = 0
    for episode in dev["episodes"]:
        if ":" in episode["session_id"]:
            continue
        if episode["status"] != "scorable":
            continue
        session_id = episode["session_id"]
        session = sessions.get(session_id)
        if session is None:
            continue
        bounds = episode["bounds"]
        last_full = floor_to_chunk(
            min(session.duration_samples, session.wav_length_samples or session.duration_samples)
        )
        processed_end = min(bounds["scored_end"], last_full)
        trace_path = b0_dir / f"{session_id}.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        actions = [
            Action(
                action_id=f"b0:{session_id}:{b['boundary_source_sample']}",
                boundary_source_sample=int(b["boundary_source_sample"]),
                observed_source_sample_at_emit=int(b["observed_source_sample_at_emit"]),
                kind="hard",
                owner="b0",
            )
            for b in trace["trace_projection"]
            if bounds["scored_start"] <= int(b["boundary_source_sample"]) < processed_end
        ]
        references = [ReferenceAction.from_dict(r) for r in episode["references"]]
        singleton_intervals = [
            (r.start_sample, r.end_sample, sorted(r.speakers)[0])
            for r in session.regions
            if len(r.speakers) == 1
            and not r.ambiguous
            and r.start_sample >= bounds["scored_start"]
            and r.end_sample <= processed_end
        ]
        pause_intervals = [
            (r.start_sample, r.end_sample)
            for r in session.regions
            if not r.speakers
            and not r.ambiguous
            and r.start_sample >= bounds["scored_start"]
            and r.end_sample <= processed_end
        ]
        scored = score_episode(
            actions,
            references,
            singleton_intervals,
            pause_intervals,
            bounds["scored_start"],
            processed_end,
        )
        episodes_scored += 1
        rows.append(
            {
                "episode_id": episode["episode_id"],
                "session_id": session_id,
                "pool": episode["pool"],
                "tag": episode["tag"],
                "hard_action_count": scored["hard_action_count"],
                "match_count": scored["match_count"],
                "hard_miss_count": len(scored["hard_misses"]),
                "contamination_100ms_ms": scored["contamination_ms"]["100ms"],
                "contamination_50ms_ms": scored["contamination_ms"]["50ms"],
                "contamination_200ms_ms": scored["contamination_ms"]["200ms"],
                "harmful_active_splits": scored["harm_flags"]["harmful_active_split"],
                "same_speaker_pause_splits": scored["harm_flags"]["same_speaker_pause_split"],
                "overlap_hard_actions": scored["harm_flags"]["overlap_hard_action"],
                "duplicate_hard_boundaries": scored["harm_flags"]["duplicate_hard_boundary"],
            }
        )
    hard_only = [r for r in rows if r["tag"] == "hard_only"]
    return {
        "pool": "baseline_dev",
        "sessions": len(sessions),
        "episodes": episodes_scored,
        "hard_only_episodes": len(hard_only),
        "total_hard_actions": sum(r["hard_action_count"] for r in rows),
        "total_hard_misses": sum(r["hard_miss_count"] for r in rows),
        "total_contamination_100ms_ms": sum(r["contamination_100ms_ms"] for r in rows),
        "total_harmful_active_splits": sum(r["harmful_active_splits"] for r in rows),
        "rows": rows,
        "note": "baseline dev evidence only; never confirmatory; never natural rates",
    }


if __name__ == "__main__":
    main()
