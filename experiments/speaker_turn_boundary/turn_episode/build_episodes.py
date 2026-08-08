"""Phase 2 bounded episode builder and manifest generator.

Implements the approved Phase 2 review bundle rev 7 (PRD Section 29, Phase 2):
bounded-episode extraction per Section 5.1, interval-valued reference construction
per Sections 6/6.7, deterministic self-hashed manifests per Section 27.3, pool split
per Section 16.4, natural-exposure window manifest, and fail-closed invariants
(29, 30, 31). Readiness (pending-start inspection) is delegated to
``state_equivalence.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schemas import ReferenceAction

CANONICAL_SAMPLE_RATE_HZ = 16000
CHUNK_SAMPLES = 512
SAMPLES_PER_MS = CANONICAL_SAMPLE_RATE_HZ // 1000

WARMUP_MS = 5000
SCORED_MS = 5000
TAIL_MS = 3000
MIN_SCORED_MS = 10_000
MAX_SCORED_MS = 20_000
MAX_FULL_MS = 30_000
LOCALIZATION_TOLERANCE_MS = 500
STABLE_INTERVAL_MS = 100
SPLIT_MIN_DISTANCE_MS = 2000
NATURAL_WINDOW_MS = 30_000

PLAN_BLOB = "24340f488f1bb46c666a5fc15eef2fc87ef1f826"

STRUCTURAL_DEFERRAL = "max_duration_and_terminal_deferred_phase3_8"

POOLS = ("diagnostic_dev", "frontier_dev")
P1_GROUP_GRAPH_HASH = "7ebf4dffa0af180910007a318d0e3d1e77f7f048dbae852199ddd45f74cce7eb"
SYNTHETIC_ORDER = ("ls_dev", "ls_held_out_clean", "ls_held_out_other", "mixed_dev_pool")
SYNTHETIC_HISTORY = {
    "ls_dev": "dev_pilot",
    "ls_held_out_clean": "held_out_pilot",
    "ls_held_out_other": "held_out_pilot",
    "mixed_dev_pool": "dev_pilot",
}


class Phase2Error(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False)


def ceil_to_chunk(sample: int) -> int:
    if sample % CHUNK_SAMPLES == 0:
        return sample
    return sample + (CHUNK_SAMPLES - sample % CHUNK_SAMPLES)


def floor_to_chunk(sample: int) -> int:
    return sample - sample % CHUNK_SAMPLES


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


@dataclass(frozen=True, slots=True)
class WindowBounds:
    warm_start: int
    scored_start: int
    scored_end: int
    tail_end: int
    unaligned_source_end: bool = False

    @property
    def scored_duration_samples(self) -> int:
        return self.scored_end - self.scored_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "warm_start": self.warm_start,
            "scored_start": self.scored_start,
            "scored_end": self.scored_end,
            "tail_end": self.tail_end,
            "unaligned_source_end": self.unaligned_source_end,
        }


def chunk_aligned_bounds(anchor_sample: int, session_end: int) -> WindowBounds:
    last_full_chunk_end = floor_to_chunk(session_end)
    unaligned = session_end % CHUNK_SAMPLES != 0
    # scored_start rounds DOWN so the scored region is always >= 10 s:
    # 10 s = 160000 samples is not a multiple of the 512-sample chunk, so
    # rounding both ends up could shrink the region below 10 s (bundle rev 8).
    scored_start = floor_to_chunk(max(0, anchor_sample - SCORED_MS * SAMPLES_PER_MS))
    scored_end = min(
        ceil_to_chunk(min(session_end, anchor_sample + SCORED_MS * SAMPLES_PER_MS)),
        last_full_chunk_end,
    )
    warm_start = floor_to_chunk(max(0, scored_start - WARMUP_MS * SAMPLES_PER_MS))
    tail_end_raw = min(session_end, scored_end + TAIL_MS * SAMPLES_PER_MS)
    tail_end = (
        min(session_end, ceil_to_chunk(tail_end_raw)) if tail_end_raw < session_end else session_end
    )
    if scored_start >= scored_end:
        raise Phase2Error(
            f"chunk_aligned_bounds produced empty scored region "
            f"({scored_start} >= {scored_end}); session_end={session_end}"
        )
    return WindowBounds(
        warm_start=warm_start,
        scored_start=scored_start,
        scored_end=scored_end,
        tail_end=tail_end,
        unaligned_source_end=unaligned,
    )


def split_candidates(
    regions: list[Any],
    window: tuple[int, int],
    targets: list[int],
) -> list[int]:
    candidates: list[int] = []
    for region in regions:
        if region.ambiguous:
            continue
        if len(region.speakers) > 1:
            continue
        duration_ms = (region.end_sample - region.start_sample) / SAMPLES_PER_MS
        if duration_ms < STABLE_INTERVAL_MS:
            continue
        boundary = region.start_sample
        if boundary <= window[0] or boundary >= window[1]:
            continue
        if any(abs(boundary - t) < SPLIT_MIN_DISTANCE_MS * SAMPLES_PER_MS for t in targets):
            continue
        candidates.append(boundary)
    return sorted(set(candidates))


def tag_episode(references: list[ReferenceAction]) -> str:
    has_overlap = any(
        r.action_kind == "soft_overlap_marker" or (r.action_kind == "unscored" and False)
        for r in references
    )
    hard = [r for r in references if r.action_kind == "hard_boundary"]
    if has_overlap:
        return "overlap_present"
    if hard:
        return "hard_only"
    return "negative_only"


@dataclass(slots=True)
class RawWord:
    speaker: str
    start_time_s: float | None
    end_time_s: float | None
    text: str
    ambiguous: bool
    path_index: int


def load_ami_raw_words(words_dir: Path, meeting_id: str) -> list[RawWord]:
    raw: list[RawWord] = []
    import xml.etree.ElementTree as ET

    for path_index, words_path in enumerate(sorted(words_dir.glob(f"{meeting_id}.*.words.xml"))):
        speaker = f"{meeting_id}.Participant{words_path.name.split('.')[1]}"
        tree = ET.parse(str(words_path))
        for element in tree.getroot().iter():
            if element.tag.lower() != "w":
                continue
            start = element.get("starttime")
            end = element.get("endtime")
            text = "".join(element.itertext()).strip()
            raw.append(
                RawWord(
                    speaker=speaker,
                    start_time_s=float(start) if start is not None else None,
                    end_time_s=float(end) if end is not None else None,
                    text=text,
                    ambiguous="%" in text,
                    path_index=path_index,
                )
            )
    return raw


def missing_timing_intervals(
    raw_words: list[RawWord],
    session_end: int,
    sample_rate_hz: int = CANONICAL_SAMPLE_RATE_HZ,
) -> list[tuple[int, int]]:
    """Frozen covering rule (bundle P2-030): each maximal run of consecutive
    missing-timing words inside one participant file is bounded deterministically
    and conservatively by its neighboring timed records in the same file: from the
    end of the previous timed word (or session start when none) to the start of the
    next timed word (or session end when none). A word without source coordinates
    cannot be proven to lie after any region boundary, so the session bounds are
    the only conservative fallback. Consecutive missing words form one covering
    interval; intervals overlapping across files are merged; spans are clamped to
    [0, session_end]."""
    if not raw_words:
        return []
    intervals: list[tuple[int, int]] = []
    for path_index in sorted({w.path_index for w in raw_words}):
        file_words = [w for w in raw_words if w.path_index == path_index]
        run_open = False
        prev_end_s = 0.0
        for word in file_words:
            if word.start_time_s is None or word.end_time_s is None:
                run_open = True
                continue
            if run_open:
                next_start_s = word.start_time_s or 0.0
                start_sample = max(0, min(session_end, int(round(prev_end_s * sample_rate_hz))))
                end_sample = max(
                    start_sample, min(session_end, int(round(next_start_s * sample_rate_hz)))
                )
                if end_sample > start_sample:
                    intervals.append((start_sample, end_sample))
                run_open = False
            prev_end_s = word.end_time_s or 0.0
        if run_open:
            start_sample = max(0, min(session_end, int(round(prev_end_s * sample_rate_hz))))
            end_sample = max(start_sample, session_end)
            if end_sample > start_sample:
                intervals.append((start_sample, end_sample))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


@dataclass(frozen=True, slots=True)
class RefSpec:
    suffix: str
    action_kind: str
    target_sample: int | None
    acceptable_interval: tuple[int, int]
    evidence_onset: int
    primary_case: bool
    gap: bool = False


def build_reference_specs(
    regions: list[Any],
    session_end: int,
    raw_words: list[RawWord] | None = None,
) -> list[RefSpec]:
    from ..ground_truth import classify_active_speaker_transitions

    specs: list[RefSpec] = []
    changes, transitions = classify_active_speaker_transitions(regions)
    onset_index: dict[int, int] = {}
    for index, region in enumerate(regions):
        onset_index.setdefault(region.start_sample, index)
    for gt_index, change in enumerate(changes):
        target = change.change_sample
        if change.kind == "clean_handoff":
            interval = (
                max(0, target - LOCALIZATION_TOLERANCE_MS * SAMPLES_PER_MS),
                target,
            )
            specs.append(RefSpec(f"gt{gt_index}", "hard_boundary", target, interval, target, True))
        elif change.kind == "gap_speaker_change":
            start = target
            b_index = onset_index.get(target)
            if b_index is not None:
                walk = b_index - 1
                while walk >= 0 and not regions[walk].speakers:
                    walk -= 1
                if walk >= 0:
                    start = regions[walk].end_sample
            if start > target:
                start = target
            specs.append(
                RefSpec(
                    f"gt{gt_index}",
                    "hard_boundary",
                    target,
                    (start, target),
                    target,
                    True,
                    gap=True,
                )
            )
        elif change.kind == "interruption_onset":
            interval = (max(0, target - LOCALIZATION_TOLERANCE_MS * SAMPLES_PER_MS), target)
            specs.append(
                RefSpec(f"gt{gt_index}", "soft_overlap_marker", target, interval, target, False)
            )
    for rank, index in enumerate(range(1, len(regions) - 1)):
        prev, current, nxt = regions[index - 1], regions[index], regions[index + 1]
        if current.speakers or not prev.speakers or not nxt.speakers:
            continue
        if prev.speakers == nxt.speakers:
            specs.append(
                RefSpec(
                    f"pause{rank}",
                    "neutral_pause",
                    None,
                    (prev.end_sample, nxt.start_sample),
                    nxt.start_sample,
                    False,
                )
            )
    departure_index = 0
    for transition in transitions:
        if transition.kind == "speaker_left":
            sample = transition.next_start_sample
            specs.append(
                RefSpec(
                    f"depart{departure_index}",
                    "state_update",
                    None,
                    (sample, sample),
                    sample,
                    False,
                )
            )
            departure_index += 1
    unscored_index = 0
    for region in regions:
        if region.ambiguous:
            specs.append(
                RefSpec(
                    f"unscored{unscored_index}",
                    "unscored",
                    None,
                    (region.start_sample, region.end_sample),
                    region.start_sample,
                    False,
                )
            )
            unscored_index += 1
    if raw_words:
        for start, end in missing_timing_intervals(raw_words, session_end):
            specs.append(
                RefSpec(
                    f"unscored{unscored_index}",
                    "unscored",
                    None,
                    (start, end),
                    start,
                    False,
                )
            )
            unscored_index += 1
    return specs


def intervals_intersect(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def references_for_episode(
    specs: list[RefSpec],
    session_id: str,
    episode_id: str,
    scored_start: int,
    processed_scored_end: int,
    pool_tag: str,
    scorable: bool,
) -> list[ReferenceAction]:
    out: list[ReferenceAction] = []
    for spec in sorted(specs, key=lambda s: (s.evidence_onset, s.suffix)):
        in_region = intervals_intersect(
            spec.acceptable_interval, (scored_start, processed_scored_end)
        )
        onset_ok = scored_start <= spec.evidence_onset < processed_scored_end
        if not in_region or not onset_ok:
            continue
        start, end = spec.acceptable_interval
        clipped = (max(start, scored_start), min(end, processed_scored_end))
        if clipped[1] <= clipped[0]:
            continue
        gap_marker = ":gap" if spec.gap else ""
        out.append(
            ReferenceAction(
                reference_id=f"{session_id}:{episode_id}:{spec.suffix}{gap_marker}",
                audio_epoch=0,
                source_session_id=session_id,
                action_kind=spec.action_kind,
                target_sample=spec.target_sample,
                acceptable_interval=clipped,
                evidence_onset_sample=spec.evidence_onset,
                scorable=scorable,
                primary_case=spec.primary_case,
                episode_pool_tag=pool_tag,
            )
        )
    out.append(
        ReferenceAction(
            reference_id=f"{session_id}:{episode_id}:structural:start",
            audio_epoch=0,
            source_session_id=session_id,
            action_kind="structural",
            target_sample=scored_start,
            acceptable_interval=(scored_start, scored_start),
            evidence_onset_sample=scored_start,
            scorable=scorable,
            primary_case=False,
            episode_pool_tag=pool_tag,
        )
    )
    out.append(
        ReferenceAction(
            reference_id=f"{session_id}:{episode_id}:structural:end",
            audio_epoch=0,
            source_session_id=session_id,
            action_kind="structural",
            target_sample=processed_scored_end,
            acceptable_interval=(processed_scored_end, processed_scored_end),
            evidence_onset_sample=processed_scored_end,
            scorable=scorable,
            primary_case=False,
            episode_pool_tag=pool_tag,
        )
    )
    return out


@dataclass(slots=True)
class SessionData:
    session_id: str
    corpus: str
    meeting_id: str
    touched_status: str
    duration_samples: int
    wav_path: str | None
    wav_abs_path: Path | None
    wav_length_samples: int | None
    wav_sha256: str | None
    annotation_sha256: str | None
    pool: str
    regions: list[Any]
    specs: list[RefSpec]
    raw_words: list[RawWord] | None = None
    historical_status: str = ""


def pool_for_session(
    session_id: str, corpus: str, by_corpus_rank: dict[str, dict[str, int]]
) -> str:
    rank = by_corpus_rank[corpus][session_id]
    return POOLS[rank % 2]


def load_session_data(
    session_id: str,
    details_row: dict[str, Any],
    corpus_root: Path,
    manifests_dir: Path,
    pilot_cases: dict[str, list[Any]],
    by_corpus_rank: dict[str, dict[str, int]],
) -> SessionData:
    corpus = str(details_row["corpus"])
    meeting_id = str(details_row.get("meeting_id") or "")
    duration_samples = int(details_row["duration_samples"])
    touched = str(details_row.get("touched_status") or "")
    historical = (
        touched if touched in ("dev_pilot", "held_out_pilot", "dev_added", "untouched") else touched
    )
    regions: list[Any]
    raw_words: list[RawWord] | None = None
    if session_id in pilot_cases:
        regions = list(pilot_cases[session_id][0].regions)
    else:
        words_dir = corpus_root / "ami" / "annotations" / "words"
        raw_words = load_ami_raw_words(words_dir, meeting_id)
        timed = [w for w in raw_words if w.start_time_s is not None and w.end_time_s is not None]
        from ..corpus.ami import AmiWord as AmiWordSchema

        words = [
            AmiWordSchema(
                speaker=w.speaker,
                start_time_s=w.start_time_s,
                end_time_s=w.end_time_s,
                text=w.text,
                ambiguous=w.ambiguous,
            )
            for w in sorted(timed, key=lambda w: (w.start_time_s or 0.0, w.end_time_s or 0.0))
        ]
        from ..corpus.ami import words_to_regions

        regions, _stats = words_to_regions(words, duration_samples)
    pool = pool_for_session(session_id, corpus, by_corpus_rank)
    specs = build_reference_specs(regions, duration_samples, raw_words)
    wav_rel = details_row.get("wav_path")
    wav_abs = (corpus_root / wav_rel).resolve() if wav_rel else None
    wav_length: int | None = None
    if wav_abs is not None and wav_abs.is_file():
        import wave as wave_module

        with wave_module.open(str(wav_abs), "rb") as handle:
            wav_length = int(handle.getnframes())
    return SessionData(
        session_id=session_id,
        corpus=corpus,
        meeting_id=meeting_id,
        touched_status=touched,
        duration_samples=duration_samples,
        wav_path=wav_rel,
        wav_abs_path=wav_abs,
        wav_length_samples=wav_length,
        wav_sha256=details_row.get("wav_sha256"),
        annotation_sha256=details_row.get("annotation_sha256"),
        pool=pool,
        regions=regions,
        specs=specs,
        raw_words=raw_words,
        historical_status=historical,
    )


def verify_targets_match_phase1(session: SessionData, details_row: dict[str, Any]) -> None:
    from .build_coverage_inventory import _classify_targets

    rebuilt = _classify_targets(session.regions)
    for key in ("hard_clean_gap_targets", "overlap_soft_targets", "same_speaker_pause_intervals"):
        expected = details_row.get(key) or []
        actual = rebuilt[key]
        if len(actual) != len(expected):
            raise Phase2Error(
                f"{session.session_id}: {key} count mismatch "
                f"(phase1={len(expected)}, rebuilt={len(actual)})"
            )
        for e, a in zip(expected, actual):
            if e != a:
                raise Phase2Error(f"{session.session_id}: {key} mismatch phase1={e} rebuilt={a}")


@dataclass(slots=True)
class EpisodeDraft:
    session_id: str
    pool: str
    anchors: list[tuple[str, int, str]]  # (typed id, sample, kind)
    bounds: WindowBounds
    tags: list[str] = field(default_factory=list)
    coverage_loss: list[dict[str, Any]] = field(default_factory=list)
    split_applied: bool = False


def _cap_scored(
    anchors: list[tuple[str, int, str]],
    bounds: WindowBounds,
) -> tuple[WindowBounds, list[dict[str, Any]]]:
    cap_end = bounds.scored_start + MAX_SCORED_MS * SAMPLES_PER_MS
    capped = WindowBounds(
        warm_start=bounds.warm_start,
        scored_start=bounds.scored_start,
        scored_end=min(bounds.scored_end, cap_end),
        tail_end=bounds.tail_end,
        unaligned_source_end=bounds.unaligned_source_end,
    )
    losses: list[dict[str, Any]] = []
    for anchor in anchors:
        if anchor[1] >= capped.scored_end:
            losses.append({"anchor": anchor[0], "reason": "scored_truncated"})
    return capped, losses


def build_episode_drafts(
    session: SessionData,
    selected_positives: list[dict[str, Any]],
    selected_negatives: list[dict[str, Any]],
) -> list[EpisodeDraft]:
    anchors: list[tuple[str, int, str]] = []
    for target in selected_positives:
        gt_index = int(target["gt_index"])
        anchors.append((f"p{gt_index}", int(target["target_sample"]), "positive"))
    for target in selected_negatives:
        rank = int(target["rank"])
        sil_start = int(target["silence_start_sample"])
        sil_end = int(target["silence_end_sample"])
        midpoint = sil_start + (sil_end - sil_start) // 2
        anchors.append((f"n{rank}", midpoint, "negative"))
    anchors.sort(key=lambda a: (a[1], a[0]))
    drafts: list[EpisodeDraft] = []
    current_anchors: list[tuple[str, int, str]] = []
    current_bounds: WindowBounds | None = None
    targets_in_window: list[int] = []

    def emit(anchors_part, bounds_part, split_applied=False, extra_loss=None):
        capped, cap_loss = _cap_scored(anchors_part, bounds_part)
        if capped.scored_end - capped.scored_start < MIN_SCORED_MS * SAMPLES_PER_MS:
            drafts.append(
                EpisodeDraft(
                    session_id=session.session_id,
                    pool=session.pool,
                    anchors=anchors_part,
                    bounds=capped,
                    coverage_loss=[
                        {"anchor": a[0], "reason": "scored_truncated"} for a in anchors_part
                    ],
                )
            )
            return
        drafts.append(
            EpisodeDraft(
                session_id=session.session_id,
                pool=session.pool,
                anchors=anchors_part,
                bounds=capped,
                coverage_loss=cap_loss + (extra_loss or []),
                split_applied=split_applied,
            )
        )

    def close_group() -> None:
        nonlocal current_anchors, current_bounds, targets_in_window
        if not current_anchors or current_bounds is None:
            current_anchors = []
            current_bounds = None
            targets_in_window = []
            return
        group_anchors = current_anchors
        group_bounds = current_bounds
        group_targets = targets_in_window
        current_anchors = []
        current_bounds = None
        targets_in_window = []
        if group_bounds.tail_end - group_bounds.warm_start <= MAX_FULL_MS * SAMPLES_PER_MS:
            emit(group_anchors, group_bounds)
            return
        candidates = split_candidates(
            session.regions, (group_bounds.warm_start, group_bounds.tail_end), group_targets
        )
        if candidates:
            split_raw = min(
                candidates,
                key=lambda c: (abs(c - (group_bounds.warm_start + group_bounds.tail_end) // 2), c),
            )
            split_at = floor_to_chunk(split_raw)
            if split_at <= group_bounds.warm_start or split_at >= group_bounds.tail_end:
                split_at = split_raw
            left_anchors = [a for a in group_anchors if a[1] < split_at]
            right_anchors = [a for a in group_anchors if a[1] >= split_at]
            if left_anchors and right_anchors:
                left_bounds = WindowBounds(
                    warm_start=group_bounds.warm_start,
                    scored_start=group_bounds.scored_start,
                    scored_end=split_at,
                    tail_end=split_at,
                    unaligned_source_end=group_bounds.unaligned_source_end,
                )
                right_bounds = WindowBounds(
                    warm_start=split_at,
                    scored_start=split_at,
                    scored_end=group_bounds.scored_end,
                    tail_end=group_bounds.tail_end,
                    unaligned_source_end=group_bounds.unaligned_source_end,
                )
                emit(left_anchors, left_bounds, split_applied=True)
                emit(right_anchors, right_bounds, split_applied=True)
                return
        kept = list(group_anchors)
        while len(kept) > 1:
            removed = kept.pop()
            last_kept = kept[-1][1]
            kept_bounds = WindowBounds(
                warm_start=group_bounds.warm_start,
                scored_start=group_bounds.scored_start,
                scored_end=min(
                    group_bounds.scored_end,
                    ceil_to_chunk(
                        min(session.duration_samples, last_kept + SCORED_MS * SAMPLES_PER_MS)
                    ),
                ),
                tail_end=group_bounds.tail_end,
                unaligned_source_end=group_bounds.unaligned_source_end,
            )
            if kept_bounds.scored_end - kept_bounds.scored_start >= MIN_SCORED_MS * SAMPLES_PER_MS:
                emit(
                    kept,
                    kept_bounds,
                    extra_loss=[{"anchor": removed[0], "reason": "no_split_point"}],
                )
                return
        emit(
            group_anchors,
            group_bounds,
            extra_loss=[{"anchor": a[0], "reason": "no_split_point"} for a in group_anchors],
        )

    for anchor in anchors:
        bounds = chunk_aligned_bounds(anchor[1], session.duration_samples)
        if current_bounds is not None and _overlaps(
            (current_bounds.warm_start, current_bounds.tail_end),
            (bounds.warm_start, bounds.tail_end),
        ):
            current_anchors.append(anchor)
            current_bounds = WindowBounds(
                warm_start=min(current_bounds.warm_start, bounds.warm_start),
                scored_start=min(current_bounds.scored_start, bounds.scored_start),
                scored_end=max(current_bounds.scored_end, bounds.scored_end),
                tail_end=max(current_bounds.tail_end, bounds.tail_end),
                unaligned_source_end=current_bounds.unaligned_source_end
                or bounds.unaligned_source_end,
            )
            targets_in_window.append(anchor[1])
            continue
        close_group()
        current_anchors = [anchor]
        current_bounds = bounds
        targets_in_window = [anchor[1]]
    close_group()
    return drafts


@dataclass(slots=True)
class EpisodeRecord:
    episode_id: str
    pool: str
    window_type: str
    status: str
    status_reason: str | None
    session_id: str
    audio_epoch: int
    bounds: WindowBounds
    tag: str | None
    anchors: list[str]
    coverage_loss: list[dict[str, Any]]
    references: list[ReferenceAction]
    historical_status: str
    wav_sha256: str | None
    annotation_sha256: str | None
    flags: dict[str, Any]
    slice_sha256: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "pool": self.pool,
            "window_type": self.window_type,
            "status": self.status,
            "status_reason": self.status_reason,
            "session_id": self.session_id,
            "audio_epoch": self.audio_epoch,
            "bounds": self.bounds.to_dict(),
            "tag": self.tag,
            "anchors": self.anchors,
            "coverage_loss": self.coverage_loss,
            "references": [r.to_dict() for r in self.references],
            "historical_status": self.historical_status,
            "wav_sha256": self.wav_sha256,
            "annotation_sha256": self.annotation_sha256,
            "flags": self.flags,
            "slice_sha256": self.slice_sha256,
        }

    def content_hash(self) -> str:
        return sha256_bytes(canonical_json(self.payload()).encode("utf-8"))


def _readiness_for(
    bounds: WindowBounds,
    wav_abs_path: Path | None,
    session_end: int,
    anchor_sample: int | None,
) -> tuple[WindowBounds, dict[str, Any]]:
    if wav_abs_path is None:
        return bounds, {"readiness": "not_applicable_no_wav"}
    from .state_equivalence import readiness_inspect

    return readiness_inspect(bounds, str(wav_abs_path), session_end, anchor_sample)


def finalize_episodes(
    session: SessionData,
    drafts: list[EpisodeDraft],
    skip_readiness: bool = False,
) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for draft in drafts:
        bounds = draft.bounds
        reason: str | None = None
        status = "scorable"
        scored_ms = (bounds.scored_end - bounds.scored_start) / SAMPLES_PER_MS
        warmup_ms = (bounds.scored_start - bounds.warm_start) / SAMPLES_PER_MS
        if scored_ms < MIN_SCORED_MS:
            reason = "scored_truncated"
        elif warmup_ms < WARMUP_MS and session.wav_path is not None:
            reason = "unstable_warmup_frontier"
        readiness_evidence: dict[str, Any] | None = None
        if status == "scorable" and reason is None and not skip_readiness:
            anchor_sample = draft.anchors[0][1] if draft.anchors else None
            final_bounds, readiness = _readiness_for(
                bounds, session.wav_abs_path, session.duration_samples, anchor_sample
            )
            bounds = final_bounds
            readiness_evidence = readiness
            if readiness.get("status") == "diagnostic_only":
                reason = str(readiness.get("reason") or "unstable_warmup_frontier")
        last_full = floor_to_chunk(
            min(session.duration_samples, session.wav_length_samples or session.duration_samples)
        )
        processed_scored_end = min(bounds.scored_end, last_full)
        anchor_out = [
            a for a in draft.anchors if not (bounds.scored_start <= a[1] < processed_scored_end)
        ]
        if anchor_out and reason is None:
            reason = "anchor_in_unprocessed_tail"
        if reason is not None:
            status = "diagnostic_only"
        tag: str | None = None
        refs_placeholder = references_for_episode(
            session.specs,
            session.session_id,
            "PLACEHOLDER",
            bounds.scored_start,
            processed_scored_end,
            "hard_only",
            True,
        )
        tag = tag_episode(refs_placeholder)
        anchor_suffix = ".".join(sorted(a[0] for a in draft.anchors))
        episode_id = f"{session.pool}:{session.session_id}:{bounds.scored_start}:{bounds.scored_end}:{anchor_suffix}"
        refs = references_for_episode(
            session.specs,
            session.session_id,
            episode_id,
            bounds.scored_start,
            processed_scored_end,
            tag,
            status == "scorable",
        )
        slice_sha: str | None = None
        if session.wav_abs_path is not None and session.wav_abs_path.is_file():
            import wave as wave_module

            with wave_module.open(str(session.wav_abs_path), "rb") as handle:
                slice_end = min(bounds.tail_end, int(handle.getnframes()))
                handle.setpos(bounds.warm_start)
                frames = handle.readframes(slice_end - bounds.warm_start)
            slice_sha = sha256_bytes(frames)
        flags: dict[str, Any] = {
            "warmup_truncated": (bounds.scored_start - bounds.warm_start) / SAMPLES_PER_MS
            < WARMUP_MS,
            "tail_truncated": (bounds.tail_end - bounds.scored_end) / SAMPLES_PER_MS < TAIL_MS,
            "unaligned_source_end": bounds.unaligned_source_end,
        }
        if readiness_evidence is not None:
            flags["readiness"] = readiness_evidence
        records.append(
            EpisodeRecord(
                episode_id=episode_id,
                pool=session.pool,
                window_type="target_enriched",
                status=status,
                status_reason=reason,
                session_id=session.session_id,
                audio_epoch=0,
                bounds=bounds,
                tag=tag,
                anchors=[a[0] for a in draft.anchors],
                coverage_loss=draft.coverage_loss,
                references=refs,
                historical_status=session.historical_status,
                wav_sha256=session.wav_sha256,
                annotation_sha256=session.annotation_sha256,
                flags=flags,
                slice_sha256=slice_sha,
            )
        )
    return records


def assert_pool_non_overlap(records: list[EpisodeRecord], pool: str) -> None:
    scored = [r for r in records if r.pool == pool and r.status == "scorable"]
    scored.sort(key=lambda r: (r.session_id, r.bounds.scored_start))
    for left, right in zip(scored, scored[1:]):
        if left.session_id != right.session_id:
            continue
        if _overlaps(
            (left.bounds.scored_start, left.bounds.scored_end),
            (right.bounds.scored_start, right.bounds.scored_end),
        ):
            raise Phase2Error(
                f"non-overlap violation in {pool}: {left.episode_id} overlaps {right.episode_id}"
            )


def synthetic_episodes(manifests_dir: Path) -> tuple[list[EpisodeRecord], dict[str, Any]]:
    from ..corpus.phase2_schemas import Phase2Manifest

    records: list[EpisodeRecord] = []
    seen: dict[tuple[str, str], str] = {}
    deduplicated: dict[str, list[str]] = {}
    excluded_real: dict[str, list[str]] = {}
    for manifest_name in SYNTHETIC_ORDER:
        manifest = Phase2Manifest.load(manifests_dir / f"{manifest_name}.json")
        skipped: list[str] = []
        for case in manifest.cases:
            wav_rel = str(case.wav_relative_path).replace("\\", "/")
            if wav_rel.startswith(("ami/", "alimeeting/")):
                excluded_real.setdefault(manifest_name, []).append(str(case.case_id))
                continue
            identity = (str(case.case_id), wav_rel, str(case.wav_sha256) if case.wav_sha256 else "")
            if identity in seen:
                skipped.append(str(case.case_id))
                continue
            seen[identity] = manifest_name
            session_id = f"{manifest_name}:{case.case_id}"
            specs = build_reference_specs(list(case.regions), int(case.duration_samples), None)
            duration = int(case.duration_samples)
            last_full = floor_to_chunk(duration)
            processed_end = min(duration, last_full)
            pool = "diagnostic_dev"
            episode_id = f"{pool}:{session_id}:0:{duration}:"
            refs = references_for_episode(
                specs, session_id, episode_id, 0, processed_end, "hard_only", True
            )
            tag = tag_episode(refs)
            refs = references_for_episode(
                specs, session_id, episode_id, 0, processed_end, tag, True
            )
            wav_sha = str(case.wav_sha256) if case.wav_sha256 else None
            records.append(
                EpisodeRecord(
                    episode_id=episode_id,
                    pool=pool,
                    window_type="target_enriched",
                    status="scorable",
                    status_reason=None,
                    session_id=session_id,
                    audio_epoch=0,
                    bounds=WindowBounds(0, 0, duration, duration),
                    tag=tag,
                    anchors=[],
                    coverage_loss=[],
                    references=refs,
                    historical_status=str(SYNTHETIC_HISTORY[manifest_name]),
                    wav_sha256=wav_sha,
                    annotation_sha256=None,
                    flags={
                        "warmup_truncated": False,
                        "tail_truncated": False,
                        "unaligned_source_end": False,
                    },
                )
            )
        if skipped:
            deduplicated[manifest_name] = skipped
    return records, {"deduplicated": deduplicated, "excluded_real_recording": excluded_real}


def natural_window_episodes(
    inventory: dict[str, Any],
    sessions_by_id: dict[str, SessionData],
    wav_sha_by_id: dict[str, str | None],
    annotation_sha_by_id: dict[str, str | None],
) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    frame = inventory["natural_exposure"]
    for window in frame["windows"]:
        session_id = str(window["session_id"])
        session = sessions_by_id.get(session_id)
        if session is None:
            continue
        if not bool(window["included"]):
            continue
        start_ms = int(window["start_ms"])
        start_sample = start_ms * SAMPLES_PER_MS
        eligible_ms = int(window["eligible_duration_ms"])
        duration_ms = session.duration_samples // SAMPLES_PER_MS
        expected_eligible = min(NATURAL_WINDOW_MS, duration_ms - start_ms)
        if eligible_ms != expected_eligible:
            raise Phase2Error(
                f"natural window {session_id}:{start_ms} frame mismatch "
                f"(frame={eligible_ms}ms, recomputed={expected_eligible}ms)"
            )
        end_sample = (start_ms + eligible_ms) * SAMPLES_PER_MS
        if end_sample > session.duration_samples:
            raise Phase2Error(f"natural window {session_id}:{start_ms} exceeds session duration")
        pool = "natural_exposure_validation"
        episode_id = f"{pool}:{session_id}:{start_sample}:{end_sample}:"
        last_full = floor_to_chunk(session.duration_samples)
        processed_end = min(end_sample, last_full)
        refs_placeholder = references_for_episode(
            session.specs, session_id, episode_id, start_sample, processed_end, "hard_only", True
        )
        tag = tag_episode(refs_placeholder)
        refs = references_for_episode(
            session.specs, session_id, episode_id, start_sample, processed_end, tag, True
        )
        records.append(
            EpisodeRecord(
                episode_id=episode_id,
                pool=pool,
                window_type="natural_exposure",
                status="scorable",
                status_reason=None,
                session_id=session_id,
                audio_epoch=0,
                bounds=WindowBounds(start_sample, start_sample, end_sample, end_sample),
                tag=tag,
                anchors=[],
                coverage_loss=[],
                references=refs,
                historical_status=session.historical_status,
                wav_sha256=wav_sha_by_id.get(session_id),
                annotation_sha256=annotation_sha_by_id.get(session_id),
                flags={
                    "warmup_truncated": True,
                    "tail_truncated": True,
                    "unaligned_source_end": session.duration_samples % CHUNK_SAMPLES != 0,
                },
            )
        )
    return records


def write_hashed_manifest(
    path: Path,
    payload: dict[str, Any],
    episode_hashes: list[tuple[str, str]],
    excluded_keys: tuple[str, ...],
) -> str:
    hashed_payload = {k: v for k, v in payload.items() if k not in excluded_keys}
    content_sha256 = sha256_bytes(canonical_json(hashed_payload).encode("utf-8"))
    payload["content_sha256"] = content_sha256
    for episode_id, _ in episode_hashes:
        for episode in payload["episodes"]:
            if episode["episode_id"] == episode_id:
                episode["episode_manifest_id"] = f"{path.stem}:{content_sha256}"
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return content_sha256


def verify_manifest(path: Path) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in raw.items() if k != "content_sha256"}
    for episode in payload.get("episodes") or []:
        episode.pop("episode_manifest_id", None)
    expected = sha256_bytes(canonical_json(payload).encode("utf-8"))
    if expected != raw["content_sha256"]:
        raise Phase2Error(f"manifest hash mismatch for {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 2 episode/reference manifests")
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="corpus root (default: STB_PHASE2_CORPORA_ROOT or TEMP/opencode/stb_phase2_corpora)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: experiments/speaker_turn_boundary/results/turn_episode_v1)",
    )
    parser.add_argument(
        "--no-readiness",
        action="store_true",
        help="skip B0 pending-start readiness inspection (diagnostic)",
    )
    args = parser.parse_args()

    from ..corpus import external
    from ..corpus.phase2_schemas import Phase2Manifest

    if args.out is None:
        out = Path(__file__).resolve().parent.parent / "results" / "turn_episode_v1"
    else:
        out = args.out
    out.mkdir(parents=True, exist_ok=True)
    corpus_root = args.corpus_root or external.corpus_root()
    manifests_dir = Path(__file__).resolve().parent.parent / "data" / "manifests"

    inventory_path = out / "coverage_inventory.json"
    details_path = out / "coverage_inventory_details.jsonl"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in inventory.items() if k != "content_sha256"}
    if sha256_bytes(canonical_json(payload).encode("utf-8")) != inventory.get("content_sha256"):
        raise Phase2Error("coverage_inventory.json content hash mismatch")

    details_rows: dict[str, dict[str, Any]] = {}
    for line in details_path.read_text(encoding="utf-8").strip().splitlines():
        row = json.loads(line)
        details_rows[str(row["session_id"])] = row

    from .build_coverage_inventory import AMI_RESERVED_ADDITIONS

    reserved_ids = {f"ami_{mid}" for mid in AMI_RESERVED_ADDITIONS}
    for session_id in details_rows:
        if session_id in reserved_ids:
            raise Phase2Error(f"reserved session present in details rows: {session_id}")

    opened_sessions = sorted(set(inventory["completed_materialized_sessions"]))
    if len(opened_sessions) != 20:
        raise Phase2Error(f"expected 20 opened sessions, found {len(opened_sessions)}")

    by_corpus_rank: dict[str, dict[str, int]] = {}
    for corpus in ("ami", "alimeeting"):
        ids = sorted(s for s in opened_sessions if str(details_rows[s]["corpus"]) == corpus)
        by_corpus_rank[corpus] = {sid: rank for rank, sid in enumerate(ids)}

    # P2-SPLIT-001: the Phase 1 group graph is frozen; the builder fails closed on a
    # changed graph hash and on any component split across pools (invariants 27, 29).
    graph = inventory["group_graph"]
    if graph.get("graph_hash") != P1_GROUP_GRAPH_HASH:
        raise Phase2Error(
            f"group graph hash mismatch (frozen={P1_GROUP_GRAPH_HASH}, "
            f"actual={graph.get('graph_hash')})"
        )
    opened_set = set(opened_sessions)
    covered_by_graph = set()
    for component_id, member_ids in graph["component_sessions"].items():
        opened_members = [sid for sid in member_ids if sid in opened_set]
        covered_by_graph.update(opened_members)
        pools = {
            pool_for_session(sid, details_rows[sid]["corpus"], by_corpus_rank)
            for sid in opened_members
        }
        if len(pools) > 1:
            raise Phase2Error(f"component split across pools: {component_id} -> {sorted(pools)}")
    if covered_by_graph != opened_set:
        raise Phase2Error(
            "opened sessions not covered by group graph components: "
            f"{sorted(opened_set - covered_by_graph)}"
        )

    pilot_cases: dict[str, list[Any]] = {}
    for name in ("ami_dev_pilot.json", "ami_held_out_pilot.json", "alimeeting_eval_pilot.json"):
        manifest = Phase2Manifest.load(manifests_dir / name)
        for case in manifest.cases:
            pilot_cases.setdefault(str(case.case_id), []).append(case)

    sessions_by_id: dict[str, SessionData] = {}
    for session_id in opened_sessions:
        row = details_rows[session_id]
        session = load_session_data(
            session_id, row, corpus_root, manifests_dir, pilot_cases, by_corpus_rank
        )
        verify_targets_match_phase1(session, row)
        sessions_by_id[session_id] = session

    records: list[EpisodeRecord] = []
    for session_id in opened_sessions:
        session = sessions_by_id[session_id]
        selection = inventory["target_enriched"]["per_session"].get(session_id) or {}
        drafts = build_episode_drafts(
            session,
            selection.get("hard_positive_selected") or [],
            selection.get("negative_selected") or [],
        )
        records.extend(finalize_episodes(session, drafts, skip_readiness=args.no_readiness))

    for pool in POOLS:
        assert_pool_non_overlap(records, pool)

    synthetic, synthetic_header = synthetic_episodes(manifests_dir)
    records.extend(synthetic)

    wav_sha_by_id = {sid: row.get("wav_sha256") for sid, row in details_rows.items()}
    annotation_sha_by_id = {sid: row.get("annotation_sha256") for sid, row in details_rows.items()}
    natural = natural_window_episodes(
        inventory, sessions_by_id, wav_sha_by_id, annotation_sha_by_id
    )

    code_hashes = {
        "build_episodes": sha256_file(Path(__file__).resolve()),
        "schemas": sha256_file(Path(__file__).resolve().parent / "schemas.py"),
        "contracts": sha256_file(Path(__file__).resolve().parent / "contracts.py"),
        "inventory": inventory["content_sha256"],
    }

    from .pinned_ledger import ledger_verification

    provenance = {
        "generated_from": code_hashes,
        **ledger_verification(),
        "group_graph_hash": graph["graph_hash"],
    }

    pool_split = {pool: [] for pool in POOLS}
    for record in records:
        pool_split.setdefault(record.pool, []).append(record.episode_id)

    dev_episodes = [r for r in records if r.pool in POOLS]
    dev_payload = {
        "schema_version": "turn_episode_v1",
        "manifest_id": "episode_manifest_dev",
        "plan_blob": PLAN_BLOB,
        **provenance,
        "pool_split": {pool: sorted(ids) for pool, ids in pool_split.items()},
        "deduplicated": synthetic_header["deduplicated"],
        "excluded_real_recording": synthetic_header["excluded_real_recording"],
        "structural_taxonomy_status": STRUCTURAL_DEFERRAL,
        "episode_count": len(dev_episodes),
        "episodes": [r.payload() for r in sorted(dev_episodes, key=lambda r: r.episode_id)],
    }
    dev_path = out / "episode_manifest_dev.json"
    write_hashed_manifest(
        dev_path, dev_payload, [(r.episode_id, "") for r in dev_episodes], ("content_sha256",)
    )
    verify_manifest(dev_path)

    sampled_ms = sum(
        int(w["eligible_duration_ms"])
        for w in inventory["natural_exposure"]["windows"]
        if str(w["session_id"]) in sessions_by_id and bool(w["included"])
    )
    natural_payload = {
        "schema_version": "turn_episode_v1",
        "manifest_id": "natural_exposure_manifest",
        "plan_blob": PLAN_BLOB,
        **provenance,
        "structural_taxonomy_status": STRUCTURAL_DEFERRAL,
        "window_frame": {
            "window_ms": inventory["natural_exposure"]["window_ms"],
            "inclusion_rule": inventory["natural_exposure"]["inclusion_rule"],
            "eligible_duration_ms": sum(
                int(w["eligible_duration_ms"])
                for w in inventory["natural_exposure"]["windows"]
                if str(w["session_id"]) in sessions_by_id
            ),
            "sampled_duration_ms": sampled_ms,
        },
        "episode_count": len(natural),
        "episodes": [r.payload() for r in sorted(natural, key=lambda r: r.episode_id)],
    }
    natural_path = out / "natural_exposure_manifest.json"
    write_hashed_manifest(
        natural_path, natural_payload, [(r.episode_id, "") for r in natural], ("content_sha256",)
    )
    verify_manifest(natural_path)

    stats = {
        "total_episodes": len(dev_episodes),
        "scorable": sum(1 for r in dev_episodes if r.status == "scorable"),
        "diagnostic_only": sum(1 for r in dev_episodes if r.status == "diagnostic_only"),
        "by_pool": {
            pool: {
                "total": sum(1 for r in dev_episodes if r.pool == pool),
                "scorable": sum(
                    1 for r in dev_episodes if r.pool == pool and r.status == "scorable"
                ),
            }
            for pool in POOLS
        },
        "by_tag": {
            tag: sum(1 for r in dev_episodes if r.status == "scorable" and r.tag == tag)
            for tag in ("hard_only", "overlap_present", "negative_only")
        },
        "reasons": {
            reason: sum(
                1
                for r in dev_episodes
                if r.status == "diagnostic_only" and r.status_reason == reason
            )
            for reason in sorted({r.status_reason for r in dev_episodes if r.status_reason})
        },
        "natural_windows": len(natural),
        "natural_sampled_ms": natural_payload["window_frame"]["sampled_duration_ms"],
    }
    print(json.dumps(stats, indent=2))
    print(f"wrote {dev_path}")
    print(f"wrote {natural_path}")


if __name__ == "__main__":
    main()
