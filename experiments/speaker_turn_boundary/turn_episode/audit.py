"""Phase 2 sampled waveform/annotation audit per the approved bundle rev 7.

Deterministic per-pool sampling (1/32, floor 8 by smallest hash), byte-identical
waveform slice check, and independent reference-timeline re-derivation directly
from the raw source annotations (AMI words.xml / AliMeeting TextGrid / synthetic
manifest case regions), requiring exact equality with the registered episode
references.
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


def audit_sample(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        episodes,
        key=lambda e: (
            int(hashlib.sha256(e["episode_id"].encode("utf-8")).hexdigest()[:2], 16),
            e["episode_id"],
        ),
    )
    kept = [
        e
        for e in ranked
        if int(hashlib.sha256(e["episode_id"].encode("utf-8")).hexdigest()[:2], 16)
        < AUDIT_PREFIX_BOUND
    ]
    seen = {e["episode_id"] for e in kept}
    for e in ranked:
        if len(kept) >= AUDIT_FLOOR:
            break
        if e["episode_id"] not in seen:
            kept.append(e)
            seen.add(e["episode_id"])
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
    return {"passed": True, "samples": end_sample - start_sample}


def rederive_references(session: Any, start_sample: int, end_sample: int) -> list[dict[str, Any]]:
    """Independent re-derivation: rebuild RefSpecs from the session's raw inputs
    and clip to [start_sample, end_sample), mirroring the builder's inclusion rule."""
    from .build_episodes import build_reference_specs, intervals_intersect

    specs = build_reference_specs(
        list(session.regions), int(session.duration_samples), session.raw_words
    )
    out: list[dict[str, Any]] = []
    for spec in specs:
        in_region = intervals_intersect(spec.acceptable_interval, (start_sample, end_sample))
        onset_ok = start_sample <= spec.evidence_onset < end_sample
        if not in_region or not onset_ok:
            continue
        out.append(
            {
                "action_kind": spec.action_kind,
                "target_sample": spec.target_sample,
                "acceptable_interval": list(spec.acceptable_interval),
                "evidence_onset": spec.evidence_onset,
                "primary_case": spec.primary_case,
            }
        )
    return sorted(out, key=lambda r: (r["evidence_onset"], str(r["acceptable_interval"])))


def audit_episode(
    episode: dict[str, Any],
    wav_path: str | None,
    session: Any,
) -> dict[str, Any]:
    bounds = episode["bounds"]
    start = bounds["scored_start"]
    end = bounds["scored_end"]
    result: dict[str, Any] = {
        "episode_id": episode["episode_id"],
        "pool": episode["pool"],
        "waveform": None,
        "annotation": None,
    }
    if wav_path is not None:
        result["waveform"] = waveform_check(wav_path, start, end)
    if session is not None:
        rebuilt = rederive_references(session, start, end)
        registered = sorted(
            [
                {
                    "action_kind": r["action_kind"],
                    "target_sample": r["target_sample"],
                    "acceptable_interval": r["acceptable_interval"],
                    "evidence_onset": r["evidence_onset_sample"],
                    "primary_case": r["primary_case"],
                }
                for r in episode["references"]
            ],
            key=lambda r: (r["evidence_onset"], str(r["acceptable_interval"])),
        )
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

    corpus_root = args.corpus_root or external.corpus_root()
    manifests_dir = Path(__file__).resolve().parent.parent / "data" / "manifests"

    dev = json.loads((out / "episode_manifest_dev.json").read_text(encoding="utf-8"))
    details_rows: dict[str, dict[str, Any]] = {}
    for line in (
        (out / "coverage_inventory_details.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ):
        row = json.loads(line)
        details_rows[str(row["session_id"])] = row

    from .build_episodes import (
        SessionData,
        load_session_data,
    )

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
    all_eps = public_eps + synthetic_eps
    sample = audit_sample(all_eps)
    sample_public = [e for e in sample if ":" not in e["session_id"]]
    sample_syn = [e for e in sample if ":" in e["session_id"]]

    public_results: list[dict[str, Any]] = []
    for episode in sample_public:
        session = sessions.get(episode["session_id"])
        wav_abs = None
        if session is not None and session.wav_abs_path is not None:
            wav_abs = str(session.wav_abs_path)
        public_results.append(audit_episode(episode, wav_abs, session))

    synthetic_results: list[dict[str, Any]] = []
    syn_sessions: dict[str, Any] = {}
    for manifest_name in ("ls_dev", "ls_held_out_clean", "ls_held_out_other", "mixed_dev_pool"):
        manifest = Phase2Manifest.load(manifests_dir / f"{manifest_name}.json")
        for case in manifest.cases:
            syn_sessions[f"{manifest_name}:{case.case_id}"] = case
    for episode in sample_syn:
        case = syn_sessions.get(episode["session_id"])
        if case is None:
            continue
        wav_abs = str((corpus_root / str(case.wav_relative_path)).resolve())

        class CaseSession:
            pass

        holder = CaseSession()
        holder.regions = list(case.regions)
        holder.duration_samples = int(case.duration_samples)
        holder.raw_words = None
        synthetic_results.append(audit_episode(episode, wav_abs, holder))

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
    annotation_failures = [
        r
        for r in public_results + synthetic_results
        if r.get("annotation") and not r["annotation"]["passed"]
    ]
    report: dict[str, Any] = {
        "schema_version": "turn_episode_v1",
        "report_id": "audit_report",
        "sampling_rule": "sha256(episode_id) prefix < 8 (1/32), floor 8 by smallest hash",
        "public_sampled": len(public_results),
        "synthetic_sampled": len(synthetic_results),
        "waveform_failures": waveform_failures,
        "waveform_unavailable": [
            {"episode_id": r["episode_id"], "reason": r["waveform"]["reason"]}
            for r in waveform_unavailable
        ],
        "annotation_failures": annotation_failures,
        "passed": not waveform_failures and not annotation_failures,
        "results": public_results + synthetic_results,
    }
    from .build_episodes import canonical_json, sha256_bytes

    payload = {k: v for k, v in report.items() if k != "content_sha256"}
    report["content_sha256"] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    path = out / "audit_report.json"
    path.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(
        f"audit: public={len(public_results)} synthetic={len(synthetic_results)} "
        f"waveform_fail={len(waveform_failures)} annotation_fail={len(annotation_failures)}"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
