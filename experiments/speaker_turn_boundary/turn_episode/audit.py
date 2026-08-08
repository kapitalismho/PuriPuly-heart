"""Phase 2 sampled waveform/annotation audit per the approved bundle rev 7-8.

Per-pool deterministic sampling (1/32, floor 8 by smallest hash per pool),
byte-identical waveform slice check against the slice SHA-256 recorded at build
time, and an INDEPENDENT reference-timeline re-derivation (its own code path, not
the builder's) directly from the raw source annotations, requiring exact equality
of reference kinds, targets, intervals, evidence onsets, scorable flags, and tags.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

AUDIT_PREFIX_BOUND = 8
AUDIT_FLOOR = 8


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
        if handle.getnchannels() != 1 or handle.getframerate() != 16000:
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
        handle.setpos(start_sample)
        frames = handle.readframes(end_sample - start_sample)
    return sha256_bytes(frames)


def independent_references(regions: list[Any], raw_words: list[Any] | None) -> list[dict[str, Any]]:
    """Independent re-derivation of the reference taxonomy.

    Deliberately implemented as its own code path (not ``build_reference_specs``):
    walks the region sequence directly, detecting clean/gap handoffs, interruption
    onsets, departures, same-speaker pauses, ambiguous/unscored spans, and (for AMI)
    missing-timing word coverage.
    """
    from ..ground_truth import classify_active_speaker_transitions

    out: list[dict[str, Any]] = []
    changes, transitions = classify_active_speaker_transitions(regions)
    onset_index: dict[int, int] = {}
    for index, region in enumerate(regions):
        onset_index.setdefault(region.start_sample, index)
    for gt_index, change in enumerate(changes):
        target = change.change_sample
        if change.kind == "clean_handoff":
            interval = (max(0, target - 8000), target)
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
        elif change.kind == "interruption_onset":
            out.append(
                {
                    "suffix": f"gt{gt_index}",
                    "action_kind": "soft_overlap_marker",
                    "target_sample": target,
                    "acceptable_interval": list((max(0, target - 8000), target)),
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
        if transition.kind == "speaker_left":
            sample = transition.next_start_sample
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
    if raw_words:
        missing = [w for w in raw_words if w.start_time_s is None or w.end_time_s is None]
        timed = [w for w in raw_words if w.start_time_s is not None and w.end_time_s is not None]
        ordered_timed = sorted(timed, key=lambda w: (w.start_time_s or 0.0, w.end_time_s or 0.0))
        session_end = max((w.end_time_s or 0.0 for w in timed), default=0.0)
        for word in missing:
            start_s = 0.0
            end_s = session_end
            for timed_word in ordered_timed:
                if (timed_word.end_time_s or 0.0) <= (
                    word.start_time_s if word.start_time_s is not None else 1e18
                ):
                    start_s = max(start_s, timed_word.end_time_s or 0.0)
                if (timed_word.start_time_s or 1e18) >= (
                    word.end_time_s if word.end_time_s is not None else 0.0
                ):
                    end_s = min(end_s, timed_word.start_time_s or 0.0)
            start_sample = max(0, int(round(start_s * 16000)))
            end_sample = max(start_sample, int(round(end_s * 16000)))
            if end_sample > start_sample:
                out.append(
                    {
                        "suffix": f"unscored{unscored_index}",
                        "action_kind": "unscored",
                        "target_sample": None,
                        "acceptable_interval": list((start_sample, end_sample)),
                        "evidence_onset": start_sample,
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


def compare_registered(
    episode: dict[str, Any],
    scored_start: int,
    processed_scored_end: int,
) -> dict[str, Any]:
    registered = []
    for r in episode["references"]:
        if r["action_kind"] == "structural":
            continue
        raw_suffix = r["reference_id"].removesuffix(":gap")
        registered.append(
            {
                "suffix": raw_suffix.rsplit(":", 1)[-1],
                "action_kind": r["action_kind"],
                "target_sample": r["target_sample"],
                "acceptable_interval": r["acceptable_interval"],
                "evidence_onset": r["evidence_onset_sample"],
                "primary_case": r["primary_case"],
                "scorable": r["scorable"],
                "gap": r["reference_id"].endswith(":gap"),
            }
        )
    return {
        "registered": sorted(
            registered, key=lambda r: (r["evidence_onset"], str(r["acceptable_interval"]))
        ),
        "processed_scored_end": processed_scored_end,
        "scored_start": scored_start,
    }


def audit_episode(
    episode: dict[str, Any],
    wav_path: str | None,
    independent_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    bounds = episode["bounds"]
    start = bounds["scored_start"]
    end = bounds["scored_end"]
    last_full = end - end % 512
    processed_end = min(end, last_full)
    result: dict[str, Any] = {
        "episode_id": episode["episode_id"],
        "pool": episode["pool"],
        "waveform": None,
        "slice_match": None,
        "annotation": None,
    }
    if wav_path is not None:
        result["waveform"] = waveform_check(wav_path, start, end)
        recorded = episode.get("slice_sha256")
        actual = slice_sha256(wav_path, start, processed_end)
        result["slice_match"] = (
            {
                "passed": recorded is not None and recorded == actual,
                "recorded": recorded,
                "actual": actual,
            }
            if actual is not None
            else {"passed": None, "reason": "wav_missing"}
        )
    rebuilt = clip_independent(independent_refs, start, processed_end)
    expected = compare_registered(episode, start, processed_end)["registered"]
    rebuilt_norm = [
        {
            "suffix": r["suffix"],
            "action_kind": r["action_kind"],
            "target_sample": r["target_sample"],
            "acceptable_interval": r["acceptable_interval"],
            "evidence_onset": r["evidence_onset"],
            "primary_case": r["primary_case"],
            "gap": r["gap"],
            "scorable": episode["status"] == "scorable",
        }
        for r in sorted(rebuilt, key=lambda r: (r["evidence_onset"], str(r["acceptable_interval"])))
    ]
    registered_norm = [
        {
            "suffix": r["suffix"],
            "action_kind": r["action_kind"],
            "target_sample": r["target_sample"],
            "acceptable_interval": r["acceptable_interval"],
            "evidence_onset": r["evidence_onset"],
            "primary_case": r["primary_case"],
            "gap": r["gap"],
            "scorable": r["scorable"],
        }
        for r in expected
    ]
    result["annotation"] = {
        "passed": rebuilt_norm == registered_norm,
        "registered_count": len(registered_norm),
        "rebuilt_count": len(rebuilt_norm),
        "mismatch": (
            None
            if rebuilt_norm == registered_norm
            else {"registered": registered_norm[:5], "rebuilt": rebuilt_norm[:5]}
        ),
    }
    result["tag_consistency"] = episode["status"] != "scorable" or all(
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
    from .build_episodes import (
        SessionData,
        canonical_json,
        load_session_data,
        sha256_bytes,
        verify_manifest,
    )

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

    public_eps = [
        e for e in dev["episodes"] if ":" not in e["session_id"] and e["status"] == "scorable"
    ]
    synthetic_eps = [e for e in dev["episodes"] if ":" in e["session_id"]]
    sample = audit_sample(public_eps + synthetic_eps)
    sample_public = [e for e in sample if ":" not in e["session_id"]]
    sample_syn = [e for e in sample if ":" in e["session_id"]]

    public_results: list[dict[str, Any]] = []
    for episode in sample_public:
        session = sessions.get(episode["session_id"])
        wav_abs = None
        if session is not None and session.wav_abs_path is not None:
            wav_abs = str(session.wav_abs_path)
        independent = independent_references(list(session.regions), session.raw_words)
        public_results.append(audit_episode(episode, wav_abs, independent))

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
        independent = independent_references(list(case.regions), None)
        synthetic_results.append(audit_episode(episode, wav_abs, independent))

    waveform_failures = [
        r
        for r in public_results + synthetic_results
        if r.get("waveform") and r["waveform"].get("passed") is False
    ]
    waveform_unavailable = [
        r
        for r in public_results + synthetic_results
        if r.get("waveform") and r["waveform"].get("passed") is None
    ]
    slice_failures = [
        r
        for r in public_results
        if r.get("slice_match") and r["slice_match"].get("passed") is False
    ]
    annotation_failures = [
        r
        for r in public_results + synthetic_results
        if r.get("annotation") and not r["annotation"]["passed"]
    ]
    tag_failures = [r for r in public_results + synthetic_results if not r.get("tag_consistency")]
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
        "sampling_rule": "per-pool sha256(episode_id) prefix < 8 (1/32), floor 8 by smallest hash",
        "public_sampled": len(public_results),
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
        and not slice_failures
        and not annotation_failures
        and not tag_failures,
        "results": public_results + synthetic_results,
    }
    payload = {k: v for k, v in report.items() if k != "content_sha256"}
    report["content_sha256"] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    path = out / "audit_report.json"
    path.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(
        f"audit: public={len(public_results)} synthetic={len(synthetic_results)} "
        f"waveform_fail={len(waveform_failures)} slice_fail={len(slice_failures)} "
        f"annotation_fail={len(annotation_failures)} tag_fail={len(tag_failures)}"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
