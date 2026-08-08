"""Build the Phase 1 metadata coverage inventory (approved Phase 1 design).

Scope (bundle rev 6, review range fef0a6b3..HEAD):
- metadata-only inventory over locally available authorized corpora (AMI, AliMeeting,
  LibriSpeech synthetic);
- deterministic B0 production-VAD baseline replay over the 12 already-materialized
  sessions (raw replay_wav_epoch boundary traces only; no coalescer);
- frozen B0-separated/B0-missed classification rule;
- frozen natural-exposure sampling frame (30 s grid, sha256 prefix < 16, computed before
  label inspection);
- frozen per-session target-enriched hash-stratified selection (12+12 caps, non-overlap);
- complete keep-together group graph with bound hash;
- per-session evidence with canonical trace-hash projection excluding
  emitted_monotonic_ns and wall-clock metadata; 12-session completeness gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.speaker_turn_boundary.config import CANONICAL_SAMPLE_RATE_HZ
from experiments.speaker_turn_boundary.corpus import external
from experiments.speaker_turn_boundary.corpus.ami import (
    AmiWord,
    _load_meetings_xml,
    _load_words_xml,
    words_to_regions,
)
from experiments.speaker_turn_boundary.corpus.phase2_schemas import Phase2Manifest
from experiments.speaker_turn_boundary.ground_truth import (
    SpeakerRegion,
    classify_active_speaker_transitions,
)
from experiments.speaker_turn_boundary.vad_baseline import replay_wav_epoch

SILERO_MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
PLAN_BLOB = "24340f488f1bb46c666a5fc15eef2fc87ef1f826"
SCHEMA_VERSION = "turn_episode_v1.coverage_inventory"
LOCALIZATION_TOLERANCE_MS_PRIMARY = 500
LOCALIZATION_TOLERANCE_MS_VIEW = 250
SAMPLES_PER_MS = CANONICAL_SAMPLE_RATE_HZ // 1000
NATURAL_WINDOW_MS = 30_000
NATURAL_HASH_PREFIX_BOUND = 16
TARGET_ENRICHED_HASH_PREFIX_BOUND = 16
MAX_POSITIVE_PER_SESSION = 12
MAX_NEGATIVE_PER_SESSION = 12

AMI_MATERIALIZED = ("ES2003a", "ES2004a", "IS1008a", "IS1009a")
AMI_DEV_TOUCHED = ("ES2003a", "IS1008a")
AMI_HELDOUT_TOUCHED = ("ES2004a", "IS1009a")
ALIMEETING_SESSIONS = (
    "R8001_M8004",
    "R8003_M8001",
    "R8007_M8010",
    "R8007_M8011",
    "R8008_M8013",
    "R8009_M8018",
    "R8009_M8019",
    "R8009_M8020",
)


class InventoryError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False)


def hash_include(session_id: str, pool: str, key: str) -> bool:
    digest = hashlib.sha256(f"{session_id}:{pool}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:2], 16) < TARGET_ENRICHED_HASH_PREFIX_BOUND


def natural_include(session_id: str, start_ms: int) -> bool:
    digest = hashlib.sha256(f"{session_id}:{start_ms}".encode("utf-8")).hexdigest()
    return int(digest[:2], 16) < NATURAL_HASH_PREFIX_BOUND


@dataclass(frozen=True, slots=True)
class SessionInventory:
    session_id: str
    corpus: str
    meeting_id: str
    touched_status: str  # dev_pilot | held_out_pilot | untouched
    duration_samples: int
    active_speech_samples: int
    wav_path: str | None
    wav_sha256: str | None
    speakers: tuple[str, ...]
    speaker_component: tuple[str, ...]
    recording_condition: str
    word_alignment_coverage: str
    annotation_file_count: int
    regions: tuple[SpeakerRegion, ...]
    hard_clean_gap_targets: list[dict[str, Any]] = field(default_factory=list)
    overlap_soft_targets: list[dict[str, Any]] = field(default_factory=list)
    same_speaker_pause_intervals: list[dict[str, Any]] = field(default_factory=list)
    short_turn_distribution_ms: dict[str, int] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "corpus": self.corpus,
            "meeting_id": self.meeting_id,
            "touched_status": self.touched_status,
            "duration_samples": self.duration_samples,
            "duration_s": round(self.duration_samples / CANONICAL_SAMPLE_RATE_HZ, 3),
            "active_speech_samples": self.active_speech_samples,
            "wav_path": self.wav_path,
            "wav_sha256": self.wav_sha256,
            "speakers": list(self.speakers),
            "speaker_component": list(self.speaker_component),
            "recording_condition": self.recording_condition,
            "word_alignment_coverage": self.word_alignment_coverage,
            "annotation_file_count": self.annotation_file_count,
            "hard_clean_gap_targets": self.hard_clean_gap_targets,
            "overlap_soft_targets": self.overlap_soft_targets,
            "same_speaker_pause_intervals": self.same_speaker_pause_intervals,
            "short_turn_distribution_ms": self.short_turn_distribution_ms,
        }


def active_speech_samples(regions: list[SpeakerRegion]) -> int:
    return sum(
        region.end_sample - region.start_sample
        for region in regions
        if region.speakers and not region.ambiguous
    )


def _target(gt_index: int, kind: str, target_sample: int, interval: list[int]) -> dict[str, Any]:
    return {
        "gt_index": gt_index,
        "kind": kind,
        "target_sample": target_sample,
        "acceptable_interval": interval,
        "evidence_onset_sample": target_sample,
    }


def _classify_targets(regions: list[SpeakerRegion]) -> dict[str, Any]:
    changes, _transitions = classify_active_speaker_transitions(regions)
    onset_index: dict[int, int] = {}
    for index, region in enumerate(regions):
        onset_index.setdefault(region.start_sample, index)
    hard_clean_gap: list[dict[str, Any]] = []
    overlap_soft: list[dict[str, Any]] = []
    same_pause: list[dict[str, Any]] = []
    for index in range(1, len(regions) - 1):
        prev, current, nxt = regions[index - 1], regions[index], regions[index + 1]
        if current.speakers or not prev.speakers or not nxt.speakers:
            continue
        if prev.speakers == nxt.speakers:
            same_pause.append(
                {
                    "speaker": sorted(prev.speakers)[0],
                    "silence_start_sample": prev.end_sample,
                    "silence_end_sample": nxt.start_sample,
                    "speech_before_start": prev.start_sample,
                    "speech_after_end": nxt.end_sample,
                }
            )
    for gt_index, change in enumerate(changes):
        target_sample = change.change_sample
        if change.kind == "clean_handoff":
            interval = [
                target_sample - LOCALIZATION_TOLERANCE_MS_PRIMARY * SAMPLES_PER_MS,
                target_sample,
            ]
            hard_clean_gap.append(_target(gt_index, "clean_handoff", target_sample, interval))
        elif change.kind == "gap_speaker_change":
            start = target_sample
            b_index = onset_index.get(target_sample)
            if b_index is not None:
                walk = b_index - 1
                while walk >= 0 and not regions[walk].speakers:
                    walk -= 1
                if walk >= 0:
                    start = regions[walk].end_sample
            if start > target_sample:
                start = target_sample
            interval = [start, target_sample]
            hard_clean_gap.append(_target(gt_index, "gap_speaker_change", target_sample, interval))
        elif change.kind == "interruption_onset":
            overlap_soft.append(
                {
                    "gt_index": gt_index,
                    "kind": "interruption_onset",
                    "target_sample": target_sample,
                }
            )
    return {
        "hard_clean_gap_targets": hard_clean_gap,
        "overlap_soft_targets": overlap_soft,
        "same_speaker_pause_intervals": same_pause,
    }


def _short_turn_distribution(regions: list[SpeakerRegion]) -> dict[str, int]:
    bins = {
        "lt250ms": 0,
        "250-500ms": 0,
        "500-750ms": 0,
        "750-1000ms": 0,
        "1000-3000ms": 0,
        "gt3000ms": 0,
    }
    for region in regions:
        if len(region.speakers) != 1:
            continue
        ms = (region.end_sample - region.start_sample) / SAMPLES_PER_MS
        if ms < 250:
            bins["lt250ms"] += 1
        elif ms < 500:
            bins["250-500ms"] += 1
        elif ms < 750:
            bins["500-750ms"] += 1
        elif ms < 1000:
            bins["750-1000ms"] += 1
        elif ms <= 3000:
            bins["1000-3000ms"] += 1
        else:
            bins["gt3000ms"] += 1
    return bins


def load_pilot_session(
    case: Any,
    wav_root: Path,
    touched_status: str,
) -> SessionInventory:
    condition = dict(case.condition or {})
    corpus = str(condition.get("corpus") or "")
    meeting_id = str(condition.get("meeting_id") or condition.get("session_id") or case.case_id)
    session_id = str(case.case_id)
    recording_condition = str(condition.get("recording_condition") or "")
    wav_path = (wav_root / case.wav_relative_path).resolve()
    wav_sha = sha256_file(wav_path) if wav_path.is_file() else None
    regions = list(case.regions)
    speakers = sorted({s for region in regions for s in region.speakers})
    partition = dict(condition.get("partition_meta") or {})
    agents = dict(partition.get("agents") or {})
    if corpus == "ami" and agents:
        component = tuple(
            sorted({agents.get(letter) for letter in speakers if agents.get(letter)})
        ) or tuple(speakers)
    else:
        component = tuple(speakers)
    targets = _classify_targets(regions)
    return SessionInventory(
        session_id=session_id,
        corpus=corpus,
        meeting_id=meeting_id,
        touched_status=touched_status,
        duration_samples=int(case.duration_samples),
        active_speech_samples=active_speech_samples(regions),
        wav_path=str(wav_path.relative_to(wav_root)) if wav_path.is_file() else None,
        wav_sha256=wav_sha,
        speakers=tuple(speakers),
        speaker_component=component,
        recording_condition=recording_condition,
        word_alignment_coverage="word_level" if corpus == "ami" else "interval_level",
        annotation_file_count=len(regions),
        regions=tuple(regions),
        hard_clean_gap_targets=targets["hard_clean_gap_targets"],
        overlap_soft_targets=targets["overlap_soft_targets"],
        same_speaker_pause_intervals=targets["same_speaker_pause_intervals"],
        short_turn_distribution_ms=_short_turn_distribution(regions),
    )


def load_ami_annotation_meetings(
    annotations_dir: Path,
    meetings_meta: dict[str, dict[str, str]],
) -> list[SessionInventory]:
    words_dir = annotations_dir / "words"
    if not words_dir.is_dir():
        return []
    by_meeting: dict[str, list[Path]] = {}
    for words_path in sorted(words_dir.glob("*.words.xml")):
        meeting_id = words_path.name.split(".")[0]
        by_meeting.setdefault(meeting_id, []).append(words_path)
    meetings: list[SessionInventory] = []
    for meeting_id, paths in sorted(by_meeting.items()):
        if meeting_id in AMI_MATERIALIZED:
            continue
        meta = meetings_meta.get(meeting_id) or {}
        duration_s = meta.get("duration_s")
        if not duration_s:
            continue
        duration_samples = int(float(duration_s) * CANONICAL_SAMPLE_RATE_HZ)
        words: list[AmiWord] = []
        for words_path in paths:
            speaker = f"{meeting_id}.Participant{words_path.name.split('.')[1]}"
            for word in _load_words_xml(words_path):
                words.append(
                    AmiWord(
                        speaker=speaker,
                        start_time_s=word.start_time_s,
                        end_time_s=word.end_time_s,
                        text=word.text,
                        ambiguous=word.ambiguous,
                    )
                )
        words.sort(key=lambda w: (w.start_time_s, w.end_time_s))
        if not words:
            continue
        regions, _ = words_to_regions(words, duration_samples)
        speakers = sorted({s for region in regions for s in region.speakers})
        agents = dict(meta.get("agents") or {})
        component = tuple(
            sorted({agents.get(letter) for letter in speakers if agents.get(letter)})
        ) or tuple(speakers)
        targets = _classify_targets(regions)
        meetings.append(
            SessionInventory(
                session_id=f"ami_annot_{meeting_id}",
                corpus="ami",
                meeting_id=meeting_id,
                touched_status="untouched",
                duration_samples=duration_samples,
                active_speech_samples=active_speech_samples(regions),
                wav_path=None,
                wav_sha256=None,
                speakers=tuple(speakers),
                speaker_component=component,
                recording_condition="annotation_only",
                word_alignment_coverage="word_level",
                annotation_file_count=len(regions),
                regions=tuple(regions),
                hard_clean_gap_targets=targets["hard_clean_gap_targets"],
                overlap_soft_targets=targets["overlap_soft_targets"],
                same_speaker_pause_intervals=targets["same_speaker_pause_intervals"],
                short_turn_distribution_ms=_short_turn_distribution(regions),
            )
        )
    return meetings


def natural_frame(sessions: list[SessionInventory]) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    eligible_ms = 0
    sampled_ms = 0
    for session in sessions:
        duration_ms = session.duration_samples // SAMPLES_PER_MS
        start_ms = 0
        while start_ms < duration_ms:
            keep = natural_include(session.session_id, start_ms)
            windows.append(
                {
                    "session_id": session.session_id,
                    "start_ms": start_ms,
                    "included": keep,
                    "eligible_duration_ms": min(NATURAL_WINDOW_MS, duration_ms - start_ms),
                }
            )
            if keep:
                sampled_ms += min(NATURAL_WINDOW_MS, duration_ms - start_ms)
            eligible_ms += min(NATURAL_WINDOW_MS, duration_ms - start_ms)
            start_ms += NATURAL_WINDOW_MS
    return {
        "window_ms": NATURAL_WINDOW_MS,
        "inclusion_rule": "sha256(session_id:start_ms) prefix < 16 (1/16)",
        "window_count": len(windows),
        "eligible_duration_ms": eligible_ms,
        "sampled_duration_ms": sampled_ms,
        "windows": windows,
    }


def _overlaps(a: list[int], b: list[int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def target_enriched_selection(sessions: list[SessionInventory]) -> dict[str, Any]:
    out: dict[str, Any] = {"per_session": {}, "eligible_counts": {}}
    for session in sessions:
        positives = session.hard_clean_gap_targets
        negatives = session.same_speaker_pause_intervals
        pos_ranked = sorted(
            enumerate(positives), key=lambda item: (item[1]["target_sample"], item[0])
        )
        neg_ranked = sorted(
            enumerate(negatives),
            key=lambda item: (
                item[1]["silence_start_sample"],
                item[1]["silence_end_sample"],
                item[0],
            ),
        )
        kept_pos: list[dict[str, Any]] = []
        for index, target in pos_ranked:
            if not hash_include(session.session_id, "positive", str(index)):
                continue
            conflict = any(
                _overlaps(target["acceptable_interval"], k["acceptable_interval"]) for k in kept_pos
            )
            if conflict:
                continue
            kept_pos.append(target)
            if len(kept_pos) >= MAX_POSITIVE_PER_SESSION:
                break
        kept_neg: list[dict[str, Any]] = []
        for index, target in neg_ranked:
            if not hash_include(session.session_id, "negative", str(index)):
                continue
            interval = [target["silence_start_sample"], target["silence_end_sample"]]
            conflict = any(_overlaps(interval, k["interval"]) for k in kept_neg)
            if conflict:
                continue
            kept_neg.append({**target, "interval": interval})
            if len(kept_neg) >= MAX_NEGATIVE_PER_SESSION:
                break
        out["eligible_counts"][session.session_id] = {
            "hard_positive_eligible": len(positives),
            "negative_eligible": len(negatives),
            "hard_positive_selected": len(kept_pos),
            "negative_selected": len(kept_neg),
        }
        out["per_session"][session.session_id] = {
            "hard_positive_selected": kept_pos,
            "negative_selected": kept_neg,
        }
    return out


def build_group_graph(sessions: list[SessionInventory]) -> dict[str, Any]:
    components: dict[tuple[str, ...], list[str]] = {}
    for session in sessions:
        key = tuple(sorted(session.speaker_component)) or (session.session_id,)
        components.setdefault(key, []).append(session.session_id)
    serialized = {"|".join(key): sorted(ids) for key, ids in sorted(components.items())}
    return {
        "component_sessions": serialized,
        "graph_hash": sha256_bytes(canonical_json(serialized).encode("utf-8")),
    }


def replay_b0(session: SessionInventory, wav_path: Path) -> dict[str, Any]:
    from puripuly_heart.core.vad.bundled import bundled_silero_vad_onnx_path
    from puripuly_heart.core.vad.silero import SileroVadOnnx

    model_path = Path(str(bundled_silero_vad_onnx_path()))
    actual = sha256_file(model_path)
    if actual != SILERO_MODEL_SHA256:
        raise InventoryError(
            f"Silero model hash mismatch: expected {SILERO_MODEL_SHA256}, got {actual}"
        )
    started = time.perf_counter()
    result = replay_wav_epoch(
        wav_path,
        audio_epoch=0,
        engine_factory=lambda: SileroVadOnnx(model_path),
    )
    elapsed_s = time.perf_counter() - started
    boundaries = list(result.boundaries)
    projected = [
        {
            "audio_epoch": b.audio_epoch,
            "boundary_source_sample": b.boundary_source_sample,
            "observed_source_sample_at_emit": b.observed_source_sample_at_emit,
            "confidence": b.confidence,
            "source": b.source,
            "debug": dict(sorted(b.debug.items(), key=lambda item: item[0])),
        }
        for b in boundaries
    ]
    trace_hash = sha256_bytes(canonical_json(projected).encode("utf-8"))
    classification: list[dict[str, Any]] = []
    for index, target in enumerate(session.hard_clean_gap_targets):
        interval = target["acceptable_interval"]
        separated = any(interval[0] <= b.boundary_source_sample <= interval[1] for b in boundaries)
        classification.append(
            {
                "gt_index": index,
                "kind": target["kind"],
                "target_sample": target["target_sample"],
                "acceptable_interval": interval,
                "b0_separated": separated,
                "b0_missed": not separated,
            }
        )
    return {
        "session_id": session.session_id,
        "corpus": session.corpus,
        "model_sha256": SILERO_MODEL_SHA256,
        "length_samples": result.length_samples,
        "boundary_count": len(boundaries),
        "trace_hash": trace_hash,
        "classification": classification,
        "b0_separated_count": sum(1 for c in classification if c["b0_separated"]),
        "b0_missed_count": sum(1 for c in classification if c["b0_missed"]),
        "trace_projection": projected,
        "runtime": {"elapsed_s": round(elapsed_s, 3)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 1 coverage inventory")
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
    parser.add_argument("--skip-b0", action="store_true", help="skip B0 replay (diagnostic)")
    args = parser.parse_args()

    corpus_root = args.corpus_root or external.corpus_root()
    if not corpus_root.is_dir():
        raise InventoryError(f"corpus root not found: {corpus_root}")
    if args.out is None:
        args.out = Path(__file__).resolve().parent.parent / "results" / "turn_episode_v1"
    args.out.mkdir(parents=True, exist_ok=True)

    wav_root = corpus_root
    manifests_dir = Path(__file__).resolve().parent.parent / "data" / "manifests"

    sessions: list[SessionInventory] = []
    for manifest_name, touched_map in (
        ("ami_dev_pilot.json", {f"ami_{c}": "dev_pilot" for c in AMI_DEV_TOUCHED}),
        ("ami_held_out_pilot.json", {f"ami_{c}": "held_out_pilot" for c in AMI_HELDOUT_TOUCHED}),
        (
            "alimeeting_eval_pilot.json",
            {f"alimeeting_{c}": "held_out_pilot" for c in ALIMEETING_SESSIONS},
        ),
    ):
        manifest = Phase2Manifest.load(manifests_dir / manifest_name)
        for case in manifest.cases:
            touched = touched_map.get(str(case.case_id), "untouched")
            sessions.append(load_pilot_session(case, wav_root, touched))

    expected_sessions = {f"ami_{mid}" for mid in AMI_MATERIALIZED} | {
        f"alimeeting_{sid}" for sid in ALIMEETING_SESSIONS
    }
    actual_sessions = {s.session_id for s in sessions}
    missing = expected_sessions - actual_sessions
    if missing:
        raise InventoryError(f"expected materialized sessions missing: {sorted(missing)}")

    annotations_dir = corpus_root / "ami" / "annotations"
    meetings_meta = _load_meetings_xml(annotations_dir) if annotations_dir.is_dir() else {}
    annotation_sessions = load_ami_annotation_meetings(annotations_dir, meetings_meta)

    synthetic_counts: dict[str, Any] = {}
    for name in ("ls_dev", "ls_held_out_clean", "ls_held_out_other", "mixed_dev_pool"):
        manifest = Phase2Manifest.load(manifests_dir / f"{name}.json")
        kinds: set[str] = set()
        for case in manifest.cases:
            for kind in (case.condition or {}).get("expected_kinds") or []:
                kinds.add(str(kind))
        synthetic_counts[name] = {
            "case_count": len(manifest.cases),
            "duration_samples": sum(c.duration_samples for c in manifest.cases),
            "active_speech_samples": sum(
                sum(
                    r.end_sample - r.start_sample
                    for r in c.regions
                    if r.speakers and not r.ambiguous
                )
                for c in manifest.cases
            ),
            "expected_kinds": sorted(kinds),
        }

    natural = natural_frame(sessions)
    enriched = target_enriched_selection(sessions)
    all_sessions = sessions + annotation_sessions
    group_graph = build_group_graph(all_sessions)

    summary: dict[str, Any] = {"corpus": {}}
    for session in all_sessions:
        entry = summary["corpus"].setdefault(
            session.corpus,
            {
                "annotated_sessions": 0,
                "materialized_sessions": 0,
                "scorable_sessions": 0,
                "hard_clean_gap_targets": 0,
                "overlap_soft_targets": 0,
                "same_speaker_pause_intervals": 0,
                "duration_s": 0.0,
                "active_speech_s": 0.0,
            },
        )
        entry["annotated_sessions"] += 1
        if session.wav_path is not None:
            entry["materialized_sessions"] += 1
            entry["scorable_sessions"] += 1
            entry["duration_s"] += session.duration_samples / CANONICAL_SAMPLE_RATE_HZ
            entry["active_speech_s"] += session.active_speech_samples / CANONICAL_SAMPLE_RATE_HZ
        entry["hard_clean_gap_targets"] += len(session.hard_clean_gap_targets)
        entry["overlap_soft_targets"] += len(session.overlap_soft_targets)
        entry["same_speaker_pause_intervals"] += len(session.same_speaker_pause_intervals)

    independent_blocks: dict[str, int] = {
        "ami": len(
            {
                tuple(sorted(s.speaker_component))
                for s in all_sessions
                if s.corpus == "ami" and s.wav_path is not None
            }
        ),
        "alimeeting": len(
            {tuple(sorted(s.speaker_component)) for s in all_sessions if s.corpus == "alimeeting"}
        ),
    }
    summary["independent_block_estimate"] = independent_blocks
    summary["untouched_scorable_sessions"] = {
        "ami": sum(
            1
            for s in all_sessions
            if s.corpus == "ami" and s.wav_path is not None and s.touched_status == "untouched"
        ),
        "alimeeting": sum(
            1 for s in all_sessions if s.corpus == "alimeeting" and s.touched_status == "untouched"
        ),
    }

    code_hashes = {
        "inventory_script": sha256_file(Path(__file__).resolve()),
        "vad_baseline": sha256_file(Path(__file__).resolve().parent.parent / "vad_baseline.py"),
        "silero_model": SILERO_MODEL_SHA256,
    }

    inventory: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "plan_blob": PLAN_BLOB,
        "corpus_root": str(corpus_root),
        "expected_materialized_sessions": sorted(expected_sessions),
        "completed_materialized_sessions": sorted(actual_sessions),
        "completeness": {"complete": missing == set(), "missing": sorted(missing)},
        "summary": summary,
        "synthetic": synthetic_counts,
        "natural_exposure": natural,
        "target_enriched": enriched,
        "group_graph": group_graph,
        "code_hashes": code_hashes,
    }
    inventory["content_sha256"] = sha256_bytes(
        canonical_json({k: v for k, v in inventory.items() if k != "content_sha256"}).encode(
            "utf-8"
        )
    )

    inventory_path = args.out / "coverage_inventory.json"
    inventory_path.write_text(canonical_json(inventory) + "\n", encoding="utf-8")

    details_path = args.out / "coverage_inventory_details.jsonl"
    with details_path.open("w", encoding="utf-8") as handle:
        for session in all_sessions:
            handle.write(json.dumps(session.to_row(), sort_keys=True) + "\n")

    b0_dir = args.out / "b0_inventory_replay"
    b0_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_b0:
        for session in sessions:
            if session.wav_path is None:
                continue
            wav_path = (wav_root / session.wav_path).resolve()
            evidence = replay_b0(session, wav_path)
            (b0_dir / f"{session.session_id}.json").write_text(
                canonical_json(evidence) + "\n", encoding="utf-8"
            )
    else:
        for session in sessions:
            if session.wav_path is None:
                continue
            target = b0_dir / f"{session.session_id}.json"
            if not target.is_file():
                raise InventoryError(f"skip-b0 requested but evidence missing: {target}")

    print(f"wrote {inventory_path}")
    print(f"wrote {details_path}")
    print(
        "completeness: "
        + json.dumps(
            {
                "expected": len(expected_sessions),
                "completed": len(actual_sessions),
                "missing": sorted(missing),
            }
        )
    )
    print("summary: " + canonical_json(summary))
    print(
        "natural: "
        + canonical_json(
            {k: natural[k] for k in ("window_count", "eligible_duration_ms", "sampled_duration_ms")}
        )
    )


if __name__ == "__main__":
    main()
