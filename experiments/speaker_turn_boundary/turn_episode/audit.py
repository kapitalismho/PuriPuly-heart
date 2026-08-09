"""Phase 2 sampled waveform/annotation audit per the approved bundle rev 7-8.

Per-pool deterministic sampling (1/32, floor 8 by smallest hash per pool),
byte-identical waveform slice check against the slice SHA-256 recorded at build
time over the FULL episode span ``[warm_start, tail_end)``, and an INDEPENDENT
reference-timeline re-derivation (its own code path, not the builder's): the raw
source annotations are parsed directly (AMI ``words.xml`` set; AliMeeting TextGrid
interval tiers), regions are derived with an independent sweep-line region builder,
transitions with an independent classifier, and references with an independent
taxonomy emission (bundle Section 11, finding P2-005; exit-gate findings
P2-AUDIT-001/002, P2-REF-004). The re-derived reference set must equal the
registered set exactly (full reference ids incl. structural refs, kinds, targets,
intervals, evidence onsets, scorable flags, episode tags, for scorable AND
diagnostic episodes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..ground_truth import SpeakerRegion

AUDIT_PREFIX_BOUND = 8
AUDIT_FLOOR = 8
LOCALIZATION_TOLERANCE_MS = 500
SAMPLES_PER_MS = 16
CANONICAL_SAMPLE_RATE_HZ = 16000

_WORDS_FILE_PATTERN = re.compile(r"^(?P<meeting>.*?)\.(?P<speaker>[A-Z])\.words\.xml$")
_SPEAKER_TIER_PATTERN = re.compile(r"^N_(?P<speaker>SPK\d+)$")
_MEETING_KEY_PATTERN = re.compile(r"^(?P<key>R\d+_M\d+)")


class AuditError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sample_episode_id(episode_id: str) -> int:
    return int(hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:2], 16)


def audit_sample(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-pool deterministic sampling with a floor of 8 per pool."""
    by_pool: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        by_pool.setdefault(episode["pool"], []).append(episode)
    kept: list[dict[str, Any]] = []
    for pool, pool_episodes in sorted(by_pool.items()):
        ranked = sorted(
            pool_episodes,
            key=lambda e: (_sample_episode_id(e["episode_id"]), e["episode_id"]),
        )
        selected = [e for e in ranked if _sample_episode_id(e["episode_id"]) < AUDIT_PREFIX_BOUND]
        seen = {e["episode_id"] for e in selected}
        for e in ranked:
            if len(selected) >= AUDIT_FLOOR:
                break
            if e["episode_id"] not in seen:
                selected.append(e)
                seen.add(e["episode_id"])
        kept.extend(selected)
    return kept


def waveform_check(wav_path: str, start_sample: int, end_sample: int) -> dict[str, Any]:
    if not Path(wav_path).is_file():
        return {"passed": None, "reason": "wav_missing"}
    import wave as wave_module

    with wave_module.open(wav_path, "rb") as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != CANONICAL_SAMPLE_RATE_HZ:
            return {"passed": False, "reason": "not_16k_mono"}
        handle.setpos(start_sample)
        frames = handle.readframes(end_sample - start_sample)
    expected = (end_sample - start_sample) * 2
    if len(frames) != expected:
        return {
            "passed": False,
            "reason": f"slice_short ({len(frames)} bytes, expected {expected})",
        }
    return {"passed": True, "samples": end_sample - start_sample, "bytes": len(frames)}


def slice_sha256(wav_path: str, start_sample: int, end_sample: int) -> str | None:
    if not Path(wav_path).is_file():
        return None
    import wave as wave_module

    with wave_module.open(wav_path, "rb") as handle:
        end_sample = min(end_sample, int(handle.getnframes()))
        handle.setpos(start_sample)
        frames = handle.readframes(end_sample - start_sample)
    return sha256_bytes(frames)


def _parse_ami_words_xml(
    words_dir: Path, meeting_id: str
) -> list[tuple[str, float | None, float | None, str, int]]:
    """Independent AMI ``words.xml`` parser (own code path; exit-gate P2-AUDIT-002).

    Returns ``(speaker, start_s, end_s, text, path_index)`` records in the same
    deterministic file order the builder uses (participant files sorted by
    participant letter; words in document order).
    """
    import xml.etree.ElementTree as ET

    records: list[tuple[str, float | None, float | None, str, int]] = []
    for path_index, words_path in enumerate(sorted(words_dir.glob(f"{meeting_id}.*.words.xml"))):
        match = _WORDS_FILE_PATTERN.match(words_path.name)
        if not match or match.group("meeting") != meeting_id:
            raise AuditError(f"unexpected AMI words filename {words_path.name}")
        speaker = f"{meeting_id}.Participant{match.group('speaker')}"
        tree = ET.parse(str(words_path))
        for element in tree.getroot().iter():
            if element.tag.lower() != "w":
                continue
            start_attr = element.get("starttime")
            end_attr = element.get("endtime")
            text = "".join(element.itertext()).strip()
            records.append(
                (
                    speaker,
                    float(start_attr) if start_attr is not None else None,
                    float(end_attr) if end_attr is not None else None,
                    text,
                    path_index,
                )
            )
    return records


def _regions_from_spans(
    spans: list[tuple[int, int, str, bool]],
    duration_samples: int,
) -> list[SpeakerRegion]:
    """Independent sweep-line region derivation from raw span records.

    Replicates the builder's region semantics (zero-length spans skipped, ends
    clamped to the session duration, contiguous boundaries, adjacent identical
    sets merged, leading/trailing silence) with its own event-sweep code path.
    """
    cleaned: list[tuple[int, int, str, bool]] = []
    boundary_set: set[int] = set()
    for start, end, speaker, ambiguous in spans:
        if end <= start:
            continue
        if start < 0 or end > duration_samples:
            end = min(end, duration_samples)
        if end <= start:
            continue
        cleaned.append((start, end, speaker, ambiguous))
        boundary_set.add(start)
        boundary_set.add(end)
    if not boundary_set:
        return [SpeakerRegion(0, 0, duration_samples, frozenset())]
    ordered = sorted(boundary_set)
    events: dict[int, list[tuple[str, bool, bool]]] = {}
    for start, end, speaker, ambiguous in cleaned:
        events.setdefault(start, []).append((speaker, ambiguous, True))
        events.setdefault(end, []).append((speaker, ambiguous, False))
    active: list[tuple[str, bool]] = []
    regions: list[SpeakerRegion] = []
    prev_sample = ordered[0]
    for sample in ordered:
        if sample > prev_sample:
            speakers = frozenset(s for s, _a in active)
            ambiguous_any = any(a for _s, a in active)
            if (
                regions
                and regions[-1].speakers == speakers
                and regions[-1].ambiguous == ambiguous_any
            ):
                regions[-1] = SpeakerRegion(
                    0, regions[-1].start_sample, sample, speakers, ambiguous_any
                )
            else:
                regions.append(SpeakerRegion(0, prev_sample, sample, speakers, ambiguous_any))
        for speaker, ambiguous, is_start in events.get(sample, []):
            if is_start:
                active.append((speaker, ambiguous))
            else:
                for i, (s, a) in enumerate(active):
                    if s == speaker and a == ambiguous:
                        del active[i]
                        break
                else:
                    for i, (s, a) in enumerate(active):
                        if s == speaker:
                            del active[i]
                            break
        prev_sample = sample
    if ordered[0] > 0:
        regions.insert(0, SpeakerRegion(0, 0, ordered[0], frozenset()))
    if ordered[-1] < duration_samples:
        regions.append(SpeakerRegion(0, ordered[-1], duration_samples, frozenset()))
    return regions


def _parse_textgrid(path: Path) -> list[tuple[str, list[tuple[float, float, str]]]]:
    """Independent TextGrid line parser (interval tiers only)."""
    tiers: list[tuple[str, list[tuple[float, float, str]]]] = []
    current_name: str | None = None
    current_class: str | None = None
    current_intervals: list[tuple[float, float, str]] = []
    interval_start: float | None = None
    interval_end: float | None = None

    def flush() -> None:
        nonlocal current_name, current_class, current_intervals
        if current_name is not None and current_class == "IntervalTier":
            tiers.append((current_name, list(current_intervals)))
        current_name = None
        current_class = None
        current_intervals = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("item ["):
            flush()
        elif line.startswith("class"):
            flush()
            current_class = _unquote(line.split("=", 1)[1].strip())
        elif line.startswith("name"):
            current_name = _unquote(line.split("=", 1)[1].strip())
        elif line.startswith("xmin"):
            interval_start = float(line.split("=", 1)[1].strip())
            interval_end = None
        elif line.startswith("xmax"):
            interval_end = float(line.split("=", 1)[1].strip())
        elif line.startswith("text") and current_name is not None:
            text = _unquote(line.split("=", 1)[1].strip())
            if interval_start is not None and interval_end is not None:
                current_intervals.append((interval_start, interval_end, text))
            interval_start = None
            interval_end = None
    flush()
    return tiers


def _unquote(value: str) -> str:
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def _audit_classify(
    regions: list[SpeakerRegion],
) -> tuple[list[Any], list[Any]]:
    """Independent transition classifier over the region sequence.

    Re-implements the frozen active-speaker transition state machine (clean
    handoff, gap speaker change, interruption onset, speaker departure,
    same-speaker pause, initial start, silence, ambiguous) with its own code
    path, returning ``(changes, transitions)`` in the same canonical order.
    """
    changes: list[Any] = []
    transitions: list[Any] = []
    if len(regions) < 2:
        return changes, transitions
    first = regions[0]
    last_active: frozenset[str] | None = None
    excluded = first.ambiguous
    if not first.ambiguous and first.speakers:
        last_active = first.speakers
    gap_pending = False
    for index in range(1, len(regions)):
        prev = regions[index - 1]
        current = regions[index]
        row = {
            "audio_epoch": current.audio_epoch,
            "prev_start_sample": prev.start_sample,
            "prev_speakers": prev.speakers,
            "next_start_sample": current.start_sample,
            "next_speakers": current.speakers,
        }
        if current.ambiguous:
            transitions.append({**row, "kind": "ambiguous", "positive": False, "ambiguous": True})
            excluded = True
            last_active = None
            gap_pending = False
            continue
        if excluded:
            transitions.append(
                {**row, "kind": "ambiguous_adjacent", "positive": False, "ambiguous": True}
            )
            excluded = False
            if current.speakers:
                last_active = current.speakers
            gap_pending = False
            continue
        if current.speakers == prev.speakers:
            if not current.speakers:
                transitions.append(
                    {**row, "kind": "silence", "positive": False, "ambiguous": False}
                )
            else:
                transitions.append(
                    {**row, "kind": "same_speaker", "positive": False, "ambiguous": False}
                )
            continue
        if not current.speakers:
            transitions.append(
                {**row, "kind": "silence_start", "positive": False, "ambiguous": False}
            )
            if last_active is not None:
                gap_pending = True
            continue
        if last_active is None:
            transitions.append(
                {**row, "kind": "initial_start", "positive": False, "ambiguous": False}
            )
            last_active = current.speakers
            gap_pending = False
            continue
        if gap_pending:
            if current.speakers == last_active:
                kind = "gap_same_speaker"
                positive = False
            else:
                kind = "gap_speaker_change"
                positive = True
            transitions.append({**row, "kind": kind, "positive": positive, "ambiguous": False})
            if positive:
                changes.append(
                    {
                        "audio_epoch": current.audio_epoch,
                        "change_sample": current.start_sample,
                        "kind": kind,
                        "prev_speakers": last_active,
                        "next_speakers": current.speakers,
                    }
                )
            last_active = current.speakers
            gap_pending = False
            continue
        new_speakers = current.speakers - last_active
        if not new_speakers:
            kind = "speaker_left"
            positive = False
        elif not (current.speakers & last_active):
            kind = "clean_handoff"
            positive = True
        else:
            kind = "interruption_onset"
            positive = True
        transitions.append({**row, "kind": kind, "positive": positive, "ambiguous": False})
        if positive:
            changes.append(
                {
                    "audio_epoch": current.audio_epoch,
                    "change_sample": current.start_sample,
                    "kind": kind,
                    "prev_speakers": last_active,
                    "next_speakers": current.speakers,
                }
            )
        last_active = current.speakers
    return changes, transitions


def _missing_timing_covering(
    words: list[tuple[str, float | None, float | None, str, int]],
    session_end: int,
) -> list[tuple[int, int]]:
    """Independent covering rule for runs of missing-timing words (P2-REF-004).

    Same frozen semantics as the builder (bundle P2-030): per participant file,
    a run of consecutive missing-timing words is bounded by the neighboring timed
    records in the same file (session start/end when none); intervals merge.
    """
    intervals: list[tuple[int, int]] = []
    for path_index in sorted({w[4] for w in words}):
        file_words = [w for w in words if w[4] == path_index]
        run_open = False
        prev_end_s = 0.0
        for _speaker, start_s, end_s, _text, _pi in file_words:
            if start_s is None or end_s is None:
                run_open = True
                continue
            if run_open:
                start_sample = max(
                    0, min(session_end, int(round(prev_end_s * CANONICAL_SAMPLE_RATE_HZ)))
                )
                end_sample = max(
                    start_sample,
                    min(session_end, int(round((start_s or 0.0) * CANONICAL_SAMPLE_RATE_HZ))),
                )
                if end_sample > start_sample:
                    intervals.append((start_sample, end_sample))
                run_open = False
            prev_end_s = end_s or 0.0
        if run_open:
            start_sample = max(
                0, min(session_end, int(round(prev_end_s * CANONICAL_SAMPLE_RATE_HZ)))
            )
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


def independent_references(
    regions: list[SpeakerRegion],
    word_records: list[tuple[str, float | None, float | None, str, int]] | None,
    session_end: int,
) -> list[dict[str, Any]]:
    """Independent reference-taxonomy emission over the region sequence.

    Returns one dict per reference with the same fields the builder's
    ``RefSpec`` carries (suffix, action_kind, target_sample, acceptable_interval,
    evidence_onset, primary_case, gap), in the same canonical order.
    """
    out: list[dict[str, Any]] = []
    changes, transitions = _audit_classify(regions)
    onset_index: dict[int, int] = {}
    for index, region in enumerate(regions):
        onset_index.setdefault(region.start_sample, index)
    for gt_index, change in enumerate(changes):
        target = int(change["change_sample"])
        kind = change["kind"]
        if kind == "clean_handoff":
            interval = (max(0, target - LOCALIZATION_TOLERANCE_MS * SAMPLES_PER_MS), target)
            out.append(
                {
                    "suffix": f"gt{gt_index}",
                    "action_kind": "hard_boundary",
                    "target_sample": target,
                    "acceptable_interval": list(interval),
                    "evidence_onset": target,
                    "primary_case": True,
                    "gap": False,
                }
            )
        elif kind == "gap_speaker_change":
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
            out.append(
                {
                    "suffix": f"gt{gt_index}",
                    "action_kind": "hard_boundary",
                    "target_sample": target,
                    "acceptable_interval": list((start, target)),
                    "evidence_onset": target,
                    "primary_case": True,
                    "gap": True,
                }
            )
        elif kind == "interruption_onset":
            interval = (max(0, target - LOCALIZATION_TOLERANCE_MS * SAMPLES_PER_MS), target)
            out.append(
                {
                    "suffix": f"gt{gt_index}",
                    "action_kind": "soft_overlap_marker",
                    "target_sample": target,
                    "acceptable_interval": list(interval),
                    "evidence_onset": target,
                    "primary_case": False,
                    "gap": False,
                }
            )
    for rank, index in enumerate(range(1, len(regions) - 1)):
        prev, current, nxt = regions[index - 1], regions[index], regions[index + 1]
        if current.speakers or not prev.speakers or not nxt.speakers:
            continue
        if prev.speakers == nxt.speakers:
            out.append(
                {
                    "suffix": f"pause{rank}",
                    "action_kind": "neutral_pause",
                    "target_sample": None,
                    "acceptable_interval": list((prev.end_sample, nxt.start_sample)),
                    "evidence_onset": nxt.start_sample,
                    "primary_case": False,
                    "gap": False,
                }
            )
    departure_index = 0
    for transition in transitions:
        if transition["kind"] == "speaker_left":
            sample = int(transition["next_start_sample"])
            out.append(
                {
                    "suffix": f"depart{departure_index}",
                    "action_kind": "state_update",
                    "target_sample": None,
                    "acceptable_interval": list((sample, sample)),
                    "evidence_onset": sample,
                    "primary_case": False,
                    "gap": False,
                }
            )
            departure_index += 1
    unscored_index = 0
    for region in regions:
        if region.ambiguous:
            out.append(
                {
                    "suffix": f"unscored{unscored_index}",
                    "action_kind": "unscored",
                    "target_sample": None,
                    "acceptable_interval": list((region.start_sample, region.end_sample)),
                    "evidence_onset": region.start_sample,
                    "primary_case": False,
                    "gap": False,
                }
            )
            unscored_index += 1
    if word_records:
        for start, end in _missing_timing_covering(word_records, session_end):
            out.append(
                {
                    "suffix": f"unscored{unscored_index}",
                    "action_kind": "unscored",
                    "target_sample": None,
                    "acceptable_interval": list((start, end)),
                    "evidence_onset": start,
                    "primary_case": False,
                    "gap": False,
                }
            )
            unscored_index += 1
    return out


def clip_independent(
    refs: list[dict[str, Any]], scored_start: int, processed_scored_end: int
) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []
    for ref in refs:
        start, end = ref["acceptable_interval"]
        if not (start < processed_scored_end and scored_start < end):
            continue
        onset = ref["evidence_onset"]
        if not (scored_start <= onset < processed_scored_end):
            continue
        new_start = max(start, scored_start)
        new_end = min(end, processed_scored_end)
        if new_end <= new_start:
            continue
        clipped.append(
            {
                **ref,
                "acceptable_interval": [new_start, new_end],
            }
        )
    return clipped


def rebuilt_reference_ids(
    clipped: list[dict[str, Any]],
    session_id: str,
    episode_id: str,
    scored_start: int,
    processed_scored_end: int,
    scorable: bool,
    tag: str,
) -> list[dict[str, Any]]:
    """Construct the full registered reference rows (ids incl. structural refs)."""
    rows: list[dict[str, Any]] = []
    for ref in clipped:
        marker = ":gap" if ref["gap"] else ""
        rows.append(
            {
                "reference_id": f"{session_id}:{episode_id}:{ref['suffix']}{marker}",
                "action_kind": ref["action_kind"],
                "target_sample": ref["target_sample"],
                "acceptable_interval": ref["acceptable_interval"],
                "evidence_onset_sample": ref["evidence_onset"],
                "scorable": scorable,
                "primary_case": ref["primary_case"],
                "episode_pool_tag": tag,
            }
        )
    rows.append(
        {
            "reference_id": f"{session_id}:{episode_id}:structural:start",
            "action_kind": "structural",
            "target_sample": scored_start,
            "acceptable_interval": [scored_start, scored_start],
            "evidence_onset_sample": scored_start,
            "scorable": scorable,
            "primary_case": False,
            "episode_pool_tag": tag,
        }
    )
    rows.append(
        {
            "reference_id": f"{session_id}:{episode_id}:structural:end",
            "action_kind": "structural",
            "target_sample": processed_scored_end,
            "acceptable_interval": [processed_scored_end, processed_scored_end],
            "evidence_onset_sample": processed_scored_end,
            "scorable": scorable,
            "primary_case": False,
            "episode_pool_tag": tag,
        }
    )
    return sorted(rows, key=lambda r: r["reference_id"])


def rebuilt_tag(
    clipped: list[dict[str, Any]],
    scored_start: int,
    processed_end: int,
    regions: list[Any],
) -> str:
    if any(r["action_kind"] == "soft_overlap_marker" for r in clipped):
        return "overlap_present"
    if any(
        not region.ambiguous
        and len(region.speakers) > 1
        and region.end_sample - region.start_sample >= 100 * SAMPLES_PER_MS
        and region.start_sample < processed_end
        and scored_start < region.end_sample
        for region in regions
    ):
        return "overlap_present"
    if any(r["action_kind"] == "hard_boundary" for r in clipped):
        return "hard_only"
    return "negative_only"


def registered_rows(episode: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in episode["references"]:
        rows.append(
            {
                "reference_id": r["reference_id"],
                "action_kind": r["action_kind"],
                "target_sample": r["target_sample"],
                "acceptable_interval": r["acceptable_interval"],
                "evidence_onset_sample": r["evidence_onset_sample"],
                "scorable": r["scorable"],
                "primary_case": r["primary_case"],
                "episode_pool_tag": r["episode_pool_tag"],
            }
        )
    return sorted(rows, key=lambda r: r["reference_id"])


def audit_episode(
    episode: dict[str, Any],
    wav_path: str | None,
    independent_refs: list[dict[str, Any]],
    regions: list[Any] | None = None,
) -> dict[str, Any]:
    bounds = episode["bounds"]
    warm_start = int(bounds["warm_start"])
    scored_start = int(bounds["scored_start"])
    scored_end = int(bounds["scored_end"])
    tail_end = int(bounds["tail_end"])
    last_full = scored_end - scored_end % 512
    processed_end = min(scored_end, last_full)
    scorable = episode["status"] == "scorable"
    result: dict[str, Any] = {
        "episode_id": episode["episode_id"],
        "pool": episode["pool"],
        "status": episode["status"],
        "waveform": None,
        "slice_match": None,
        "annotation": None,
    }
    if wav_path is not None:
        result["waveform"] = waveform_check(wav_path, warm_start, tail_end)
        recorded = episode.get("slice_sha256")
        if recorded is not None:
            actual = slice_sha256(wav_path, warm_start, tail_end)
            result["slice_match"] = (
                {
                    "passed": recorded == actual,
                    "recorded": recorded,
                    "actual": actual,
                    "span": [warm_start, tail_end],
                }
                if actual is not None
                else {"passed": None, "reason": "wav_missing"}
            )
        else:
            result["slice_match"] = {"passed": None, "reason": "no_recorded_slice"}
    clipped = clip_independent(independent_refs, scored_start, processed_end)
    rebuilt = rebuilt_reference_ids(
        clipped,
        episode["session_id"],
        episode["episode_id"],
        scored_start,
        processed_end,
        scorable,
        episode["tag"],
    )
    registered = registered_rows(episode)
    result["annotation"] = {
        "passed": rebuilt == registered,
        "registered_count": len(registered),
        "rebuilt_count": len(rebuilt),
        "mismatch": (
            None
            if rebuilt == registered
            else {"registered": registered[:5], "rebuilt": rebuilt[:5]}
        ),
    }
    rebuilt_tag_value = rebuilt_tag(clipped, scored_start, processed_end, regions or [])
    result["tag_consistency"] = rebuilt_tag_value == episode["tag"] and all(
        r["episode_pool_tag"] == episode["tag"] for r in episode["references"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 sampled audit")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: results/turn_episode_v1)",
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

    from ..corpus import external
    from ..corpus.phase2_schemas import Phase2Manifest
    from .build_episodes import canonical_json, sha256_bytes, verify_manifest
    from .pinned_ledger import ledger_verification

    corpus_root = args.corpus_root or external.corpus_root()
    manifests_dir = Path(__file__).resolve().parent.parent / "data" / "manifests"
    dev_path = out / "episode_manifest_dev.json"
    verify_manifest(dev_path)
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    details_rows: dict[str, dict[str, Any]] = {}
    for line in (
        (out / "coverage_inventory_details.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ):
        row = json.loads(line)
        details_rows[str(row["session_id"])] = row

    public_eps = [
        e for e in dev["episodes"] if ":" not in e["session_id"] and e["status"] == "scorable"
    ]
    diagnostic_eps = [
        e for e in dev["episodes"] if ":" not in e["session_id"] and e["status"] != "scorable"
    ]
    synthetic_eps = [e for e in dev["episodes"] if ":" in e["session_id"]]
    sample = audit_sample(public_eps + synthetic_eps)
    sample_public = [e for e in sample if ":" not in e["session_id"]]
    sample_syn = [e for e in sample if ":" in e["session_id"]]
    sample_diag = audit_sample(diagnostic_eps) if diagnostic_eps else []

    public_results: list[dict[str, Any]] = []
    for episode in sample_public:
        session_id = episode["session_id"]
        row = details_rows.get(session_id)
        wav_abs = None
        independent: list[dict[str, Any]] = []
        regions: list[Any] = []
        if row is not None and row.get("wav_path"):
            wav_abs = str((corpus_root / row["wav_path"]).resolve())
        if row is not None:
            corpus = str(row["corpus"])
            duration_samples = int(row["duration_samples"])
            if corpus == "ami":
                words_dir = corpus_root / "ami" / "annotations" / "words"
                word_records = _parse_ami_words_xml(words_dir, str(row.get("meeting_id") or ""))
                spans = [
                    (
                        int(round(start * CANONICAL_SAMPLE_RATE_HZ)),
                        int(round(end * CANONICAL_SAMPLE_RATE_HZ)),
                        speaker,
                        "%" in text,
                    )
                    for speaker, start, end, text, _pi in word_records
                    if start is not None and end is not None
                ]
                regions = _regions_from_spans(spans, duration_samples)
                independent = independent_references(regions, word_records, duration_samples)
            else:
                textgrid_dir = (
                    corpus_root / "alimeeting" / "Eval_Ali" / "Eval_Ali_far" / "textgrid_dir"
                )
                meeting_key = _MEETING_KEY_PATTERN.match(str(row.get("meeting_id") or "")).group(
                    "key"
                )
                tiers = _parse_textgrid(textgrid_dir / f"{meeting_key}.TextGrid")
                spans = []
                for tier_name, tier_intervals in tiers:
                    tier_match = _SPEAKER_TIER_PATTERN.match(tier_name)
                    speaker = tier_match.group("speaker") if tier_match else tier_name
                    for start, end, text in tier_intervals:
                        if not text.strip():
                            continue
                        spans.append(
                            (
                                int(round(start * CANONICAL_SAMPLE_RATE_HZ)),
                                int(round(end * CANONICAL_SAMPLE_RATE_HZ)),
                                speaker,
                                False,
                            )
                        )
                regions = _regions_from_spans(spans, duration_samples)
                independent = independent_references(regions, None, duration_samples)
        public_results.append(audit_episode(episode, wav_abs, independent, regions))

    diag_results: list[dict[str, Any]] = []
    for episode in sample_diag:
        session_id = episode["session_id"]
        row = details_rows.get(session_id)
        wav_abs = None
        independent: list[dict[str, Any]] = []
        regions: list[Any] = []
        if row is not None and row.get("wav_path"):
            wav_abs = str((corpus_root / row["wav_path"]).resolve())
        if row is not None:
            corpus = str(row["corpus"])
            duration_samples = int(row["duration_samples"])
            if corpus == "ami":
                words_dir = corpus_root / "ami" / "annotations" / "words"
                word_records = _parse_ami_words_xml(words_dir, str(row.get("meeting_id") or ""))
                spans = [
                    (
                        int(round(start * CANONICAL_SAMPLE_RATE_HZ)),
                        int(round(end * CANONICAL_SAMPLE_RATE_HZ)),
                        speaker,
                        "%" in text,
                    )
                    for speaker, start, end, text, _pi in word_records
                    if start is not None and end is not None
                ]
                regions = _regions_from_spans(spans, duration_samples)
                independent = independent_references(regions, word_records, duration_samples)
            else:
                textgrid_dir = (
                    corpus_root / "alimeeting" / "Eval_Ali" / "Eval_Ali_far" / "textgrid_dir"
                )
                meeting_key = _MEETING_KEY_PATTERN.match(str(row.get("meeting_id") or "")).group(
                    "key"
                )
                tiers = _parse_textgrid(textgrid_dir / f"{meeting_key}.TextGrid")
                spans = []
                for tier_name, tier_intervals in tiers:
                    tier_match = _SPEAKER_TIER_PATTERN.match(tier_name)
                    speaker = tier_match.group("speaker") if tier_match else tier_name
                    for start, end, text in tier_intervals:
                        if not text.strip():
                            continue
                        spans.append(
                            (
                                int(round(start * CANONICAL_SAMPLE_RATE_HZ)),
                                int(round(end * CANONICAL_SAMPLE_RATE_HZ)),
                                speaker,
                                False,
                            )
                        )
                regions = _regions_from_spans(spans, duration_samples)
                independent = independent_references(regions, None, duration_samples)
        diag_results.append(audit_episode(episode, wav_abs, independent, regions))

    syn_sessions: dict[str, Any] = {}
    for manifest_name in ("ls_dev", "ls_held_out_clean", "ls_held_out_other", "mixed_dev_pool"):
        manifest = Phase2Manifest.load(manifests_dir / f"{manifest_name}.json")
        for case in manifest.cases:
            syn_sessions[f"{manifest_name}:{case.case_id}"] = case
    synthetic_results: list[dict[str, Any]] = []
    for episode in sample_syn:
        case = syn_sessions.get(episode["session_id"])
        if case is None:
            continue
        wav_abs = str((corpus_root / str(case.wav_relative_path)).resolve())
        independent = independent_references(list(case.regions), None, case.duration_samples)
        synthetic_results.append(audit_episode(episode, wav_abs, independent, list(case.regions)))

    all_results = public_results + diag_results + synthetic_results
    waveform_failures = [
        r for r in all_results if r.get("waveform") and r["waveform"].get("passed") is False
    ]
    waveform_unavailable = [
        r for r in all_results if r.get("waveform") and r["waveform"].get("passed") is None
    ]
    slice_failures = [
        r for r in all_results if r.get("slice_match") and r["slice_match"].get("passed") is False
    ]
    annotation_failures = [
        r for r in all_results if r.get("annotation") and not r["annotation"]["passed"]
    ]
    tag_failures = [r for r in all_results if not r.get("tag_consistency")]
    report: dict[str, Any] = {
        "schema_version": "turn_episode_v1",
        "report_id": "audit_report",
        "structural_taxonomy_status": "max_duration_and_terminal_deferred_phase3_8",
        "generated_from": {
            "audit": sha256_bytes(Path(__file__).resolve().read_bytes()),
            "build_episodes": sha256_bytes(
                (Path(__file__).resolve().parent / "build_episodes.py").read_bytes()
            ),
            "schemas": sha256_bytes((Path(__file__).resolve().parent / "schemas.py").read_bytes()),
            "episode_manifest_dev": dev.get("content_sha256"),
        },
        **ledger_verification(),
        "sampling_rule": "per-pool sha256(episode_id) prefix < 8 (1/32), floor 8 by smallest hash",
        "public_sampled": len(public_results),
        "diagnostic_sampled": len(diag_results),
        "synthetic_sampled": len(synthetic_results),
        "waveform_failures": waveform_failures,
        "waveform_unavailable": [
            {"episode_id": r["episode_id"], "reason": r["waveform"]["reason"]}
            for r in waveform_unavailable
        ],
        "slice_failures": slice_failures,
        "annotation_failures": annotation_failures,
        "tag_failures": tag_failures,
        "passed": not waveform_failures
        and not waveform_unavailable
        and not slice_failures
        and not annotation_failures
        and not tag_failures,
        "results": all_results,
    }
    payload = {k: v for k, v in report.items() if k != "content_sha256"}
    report["content_sha256"] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    path = out / "audit_report.json"
    path.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(
        f"audit: public={len(public_results)} diag={len(diag_results)} "
        f"synthetic={len(synthetic_results)} waveform_fail={len(waveform_failures)} "
        f"slice_fail={len(slice_failures)} annotation_fail={len(annotation_failures)} "
        f"tag_fail={len(tag_failures)}"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
