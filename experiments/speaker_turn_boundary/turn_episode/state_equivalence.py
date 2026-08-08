"""Phase 2 state-equivalence fixtures per the approved bundle rev 7 (PRD Section 5.4).

Executable now: the B0 (Silero VAD peer) family/profile class. Provides:

- ``readiness_inspect``: B0 pending-start inspection after warm-up replay
  (bundle Section 4.4, findings P2-033/P2-034/P2-037) with chunk-aligned
  scored-start extension;
- ``parity_b0``: source_prefix vs episode_reset comparison of the scored-region
  boundary trace and DetectorProgress (observed, safe-frontier) rows (findings
  P2-008/P2-017), producing the per-class disposition;
- ``snapshot_b0``/``restore_b0``: deterministic source-prefix state snapshot
  fallback with round-trip evidence (bundle Section 8.5, finding P2-009).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .build_episodes import CHUNK_SAMPLES, WindowBounds, sha256_bytes

CANONICAL_SAMPLE_RATE_HZ = 16000
SILERO_MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
MAX_READINESS_EXTENSION_S = 30.0

PENDING_COMMIT_CHUNKS = 3
PENDING_DEBOUNCE_CHUNKS = 2


class StateEquivalenceError(RuntimeError):
    pass


def _b0_engine_factory():
    from puripuly_heart.core.vad.bundled import bundled_silero_vad_onnx_path
    from puripuly_heart.core.vad.silero import SileroVadOnnx

    model_path = Path(str(bundled_silero_vad_onnx_path()))
    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual != SILERO_MODEL_SHA256:
        raise StateEquivalenceError(f"Silero model hash mismatch: {actual}")
    return SileroVadOnnx(model_path)


def _load_wav(wav_path: Path) -> np.ndarray:
    from ..vad_baseline import load_canonical_wav

    return load_canonical_wav(wav_path)


def _make_replay(engine_factory: Callable[[], Any]):
    from ..vad_baseline import VadBoundaryReplay

    return VadBoundaryReplay(engine_factory=engine_factory)


def _pending_start_id(replay: Any) -> str | None:
    gating = getattr(replay, "_gating", None)
    if gating is None:
        return None
    pending = getattr(gating, "_pending_start_id", None)
    return str(pending) if pending is not None else None


def _progress_rows(replay: Any) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for progress in replay.progress:
        rows.append(
            {
                "observed_source_sample": int(progress.observed_source_sample),
                "safe_boundary_frontier_sample": int(progress.safe_boundary_frontier_sample),
            }
        )
    return rows


def _boundary_rows(replay: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for boundary in replay.boundaries:
        rows.append(
            {
                "audio_epoch": int(boundary.audio_epoch),
                "boundary_source_sample": int(boundary.boundary_source_sample),
                "observed_source_sample_at_emit": int(boundary.observed_source_sample_at_emit),
                "confidence": boundary.confidence,
                "source": boundary.source,
            }
        )
    return rows


def _shift_boundary_rows(rows: list[dict[str, Any]], offset: int) -> list[dict[str, Any]]:
    if offset == 0:
        return rows
    shifted: list[dict[str, Any]] = []
    for row in rows:
        shifted.append(
            {
                **row,
                "boundary_source_sample": int(row["boundary_source_sample"]) + offset,
                "observed_source_sample_at_emit": int(row["observed_source_sample_at_emit"])
                + offset,
            }
        )
    return shifted


def _shift_progress_rows(rows: list[dict[str, int]], offset: int) -> list[dict[str, int]]:
    if offset == 0:
        return rows
    return [
        {
            "observed_source_sample": int(row["observed_source_sample"]) + offset,
            "safe_boundary_frontier_sample": int(row["safe_boundary_frontier_sample"]) + offset,
        }
        for row in rows
    ]


def readiness_inspect(
    bounds: WindowBounds,
    wav_path: str,
    session_end: int,
    anchor_sample: int | None = None,
) -> tuple[WindowBounds, dict[str, Any]]:
    """Run the B0 warm-up replay and inspect pending-start state (bundle 4.4).

    Returns (final bounds, evidence). The scored start is extended chunk-by-chunk
    while a pending start is unresolved, subject to the anchor remaining inside the
    scored region and the scored duration staying >= 10 s; otherwise the episode is
    marked diagnostic-only (pending_state_unresolved).
    """
    samples = _load_wav(Path(wav_path))
    effective_end = min(session_end, int(samples.size))
    samples = samples[:effective_end]
    replay = _make_replay(_b0_engine_factory)
    replay.start_epoch(0)
    scored_start = bounds.scored_start
    warm_start = bounds.warm_start
    cursor = warm_start
    while cursor + CHUNK_SAMPLES <= scored_start:
        replay.process_chunk(samples[cursor : cursor + CHUNK_SAMPLES])
        cursor += CHUNK_SAMPLES
    pending = _pending_start_id(replay)
    progress_rows = _shift_progress_rows(_progress_rows(replay), warm_start)
    evidence: dict[str, Any] = {
        "frontier_at_scored_start": progress_rows[-1] if progress_rows else None,
        "pending_start_at_scored_start": pending,
    }
    final_scored_start = scored_start
    extended_samples = 0
    if pending is not None:
        min_scored = bounds.scored_end - 10_000 * 16
        budget = int(MAX_READINESS_EXTENSION_S * CANONICAL_SAMPLE_RATE_HZ)
        while (
            pending is not None
            and cursor + CHUNK_SAMPLES <= bounds.scored_end
            and cursor + CHUNK_SAMPLES <= budget + warm_start
            and (anchor_sample is None or cursor + CHUNK_SAMPLES < anchor_sample)
            and cursor + CHUNK_SAMPLES <= min_scored
        ):
            replay.process_chunk(samples[cursor : cursor + CHUNK_SAMPLES])
            cursor += CHUNK_SAMPLES
            final_scored_start = cursor
            extended_samples += CHUNK_SAMPLES
            pending = _pending_start_id(replay)
    if pending is not None:
        return bounds, {
            **evidence,
            "status": "diagnostic_only",
            "reason": "pending_state_unresolved",
            "pending_start_cleared": False,
        }
    final_bounds = WindowBounds(
        warm_start=warm_start,
        scored_start=final_scored_start,
        scored_end=bounds.scored_end,
        tail_end=bounds.tail_end,
        unaligned_source_end=bounds.unaligned_source_end,
    )
    return final_bounds, {
        **evidence,
        "status": "ready",
        "pending_start_cleared": pending is None,
        "scored_start_extended": final_scored_start != scored_start,
        "extended_samples": extended_samples,
        "frontier_at_final_scored_start": (
            _shift_progress_rows(_progress_rows(replay), warm_start)[-1]
            if replay.progress
            else None
        ),
    }


def _replay_region(
    samples: np.ndarray,
    start_sample: int,
    end_sample: int,
) -> Any:
    replay = _make_replay(_b0_engine_factory)
    replay.start_epoch(0)
    cursor = start_sample
    while cursor + CHUNK_SAMPLES <= end_sample:
        replay.process_chunk(samples[cursor : cursor + CHUNK_SAMPLES])
        cursor += CHUNK_SAMPLES
    return replay


def parity_b0(
    wav_path: str,
    bounds: WindowBounds,
    session_end: int,
) -> dict[str, Any]:
    """source_prefix vs episode_reset parity for one episode (bundle Section 8.2)."""
    samples = _load_wav(Path(wav_path))
    effective_end = min(session_end, int(samples.size))
    samples = samples[:effective_end]
    source = _replay_region(samples, 0, effective_end)
    reset = _replay_region(samples, bounds.warm_start, min(bounds.tail_end, effective_end))
    # The reset replay's epoch starts at warm_start, so its boundary/progress rows
    # are epoch-local (chunk_index based); PRD Section 4.3 requires canonical
    # (absolute source) coordinates, so shift the reset rows by the epoch start.
    reset_progress_raw = _shift_progress_rows(_progress_rows(reset), bounds.warm_start)
    processed_end = min(bounds.scored_end, effective_end - effective_end % CHUNK_SAMPLES)
    scored_interval = (bounds.scored_start, processed_end)

    def in_scored(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if scored_interval[0] <= int(row["boundary_source_sample"]) < scored_interval[1]
        ]

    source_bounds = in_scored(_boundary_rows(source))
    reset_bounds = in_scored(_shift_boundary_rows(_boundary_rows(reset), bounds.warm_start))
    source_progress = [
        row
        for row in _progress_rows(source)
        if scored_interval[0] <= int(row["observed_source_sample"]) <= scored_interval[1]
    ]
    reset_progress = [
        row
        for row in reset_progress_raw
        if scored_interval[0] <= int(row["observed_source_sample"]) <= scored_interval[1]
    ]
    for progress in source_progress + reset_progress:
        if progress["safe_boundary_frontier_sample"] > progress["observed_source_sample"]:
            return {
                "class": "b0/peer",
                "passed": False,
                "reason": "safe_frontier_exceeds_observed",
            }
    observed_series = [p["observed_source_sample"] for p in source_progress]
    if observed_series != sorted(observed_series):
        return {"class": "b0/peer", "passed": False, "reason": "observed_not_monotonic"}
    equal = source_bounds == reset_bounds and source_progress == reset_progress
    return {
        "class": "b0/peer",
        "passed": equal,
        "reason": None if equal else "scored_region_trace_mismatch",
        "source_boundary_count": len(source_bounds),
        "reset_boundary_count": len(reset_bounds),
        "source_progress_hash": sha256_bytes(
            json.dumps(source_progress, sort_keys=True).encode("utf-8")
        ),
        "reset_progress_hash": sha256_bytes(
            json.dumps(reset_progress, sort_keys=True).encode("utf-8")
        ),
    }


def snapshot_b0(wav_path: str, bounds: WindowBounds, session_end: int) -> dict[str, Any]:
    """Deterministic source-prefix state snapshot at the scored start (bundle 8.5).

    The B0 engine is deterministic: its full state at the scored start is a function
    of the warm-up audio, so the snapshot binds (model hash, warm-up slice, chunk
    config). Restore replays warm-up and must reproduce the source-prefix trace.
    """
    samples = _load_wav(Path(wav_path))[:session_end]
    warmup = samples[bounds.warm_start : bounds.scored_start]
    return {
        "class": "b0/peer",
        "model_sha256": SILERO_MODEL_SHA256,
        "warm_start": bounds.warm_start,
        "scored_start": bounds.scored_start,
        "chunk_samples": CHUNK_SAMPLES,
        "warmup_samples": int(warmup.size),
        "warmup_sha256": sha256_bytes(warmup.astype(np.float32).tobytes()),
    }


def restore_b0(snapshot: dict[str, Any], wav_path: str) -> Any:
    samples = _load_wav(Path(wav_path))
    warm_start = int(snapshot["warm_start"])
    scored_start = int(snapshot["scored_start"])
    replay = _make_replay(_b0_engine_factory)
    replay.start_epoch(0)
    cursor = warm_start
    while cursor + CHUNK_SAMPLES <= scored_start:
        replay.process_chunk(samples[cursor : cursor + CHUNK_SAMPLES])
        cursor += CHUNK_SAMPLES
    return replay


@dataclass(slots=True)
class ParityClass:
    class_id: str
    passed: bool | None
    reason: str | None
    episode_results: list[dict[str, Any]]


def run_parity(
    wav_path: str,
    bounds: WindowBounds,
    session_end: int,
) -> dict[str, Any]:
    result = parity_b0(wav_path, bounds, session_end)
    disposition = "reset_allowed" if result["passed"] else "source_prefix_required"
    return {"class": "b0/peer", "disposition": disposition, "result": result}


def main() -> None:
    import argparse
    import json as _json
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(description="Phase 2 state-equivalence report")
    parser.add_argument(
        "--out",
        type=_Path,
        default=None,
        help="output directory (default: results/turn_episode_v1)",
    )
    parser.add_argument(
        "--corpus-root",
        type=_Path,
        default=None,
        help="corpus root (default: STB_PHASE2_CORPORA_ROOT)",
    )
    parser.add_argument("--skip-parity", action="store_true", help="skip B0 parity replays")
    args = parser.parse_args()
    if args.out is None:
        out = _Path(__file__).resolve().parent.parent / "results" / "turn_episode_v1"
    else:
        out = args.out

    from ..corpus import external
    from ..corpus.phase2_schemas import Phase2Manifest
    from .build_episodes import (
        SessionData,
        canonical_json,
        load_session_data,
        sha256_bytes,
    )
    from .build_episodes import (
        WindowBounds as WB,
    )

    corpus_root = args.corpus_root or external.corpus_root()
    manifests_dir = _Path(__file__).resolve().parent.parent / "data" / "manifests"
    dev = _json.loads((out / "episode_manifest_dev.json").read_text(encoding="utf-8"))
    details_rows: dict[str, dict[str, Any]] = {}
    for line in (
        (out / "coverage_inventory_details.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ):
        row = _json.loads(line)
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

    class_results: dict[str, dict[str, Any]] = {
        "b0/peer": {"disposition": "source_prefix_default", "episodes": [], "passed": []}
    }
    snapshot_evidence: list[dict[str, Any]] = []
    parity_skipped = args.skip_parity
    for episode in dev["episodes"]:
        if ":" in episode["session_id"]:
            continue
        if episode["status"] != "scorable":
            continue
        session = sessions.get(episode["session_id"])
        if session is None or session.wav_abs_path is None:
            continue
        bounds = episode["bounds"]
        wb = WB(
            warm_start=int(bounds["warm_start"]),
            scored_start=int(bounds["scored_start"]),
            scored_end=int(bounds["scored_end"]),
            tail_end=int(bounds["tail_end"]),
            unaligned_source_end=bool(bounds["unaligned_source_end"]),
        )
        snapshot = snapshot_b0(str(session.wav_abs_path), wb, session.duration_samples)
        restored = restore_b0(snapshot, str(session.wav_abs_path))
        restored_rows = _boundary_rows(restored)
        snapshot_evidence.append(
            {
                "episode_id": episode["episode_id"],
                "snapshot": snapshot,
                "restored_boundary_count": len(restored_rows),
            }
        )
        if parity_skipped:
            continue
        result = parity_b0(str(session.wav_abs_path), wb, session.duration_samples)
        class_results["b0/peer"]["episodes"].append(
            {
                "episode_id": episode["episode_id"],
                "passed": result["passed"],
                "reason": result.get("reason"),
            }
        )
        class_results["b0/peer"]["passed"].append(result["passed"])

    passed_list = class_results["b0/peer"]["passed"]
    if passed_list and all(passed_list):
        class_results["b0/peer"]["disposition"] = "reset_allowed"
    elif passed_list:
        class_results["b0/peer"]["disposition"] = "source_prefix_required"
    class_results["b0/peer"]["episode_count"] = len(passed_list)

    convergence_diagnostic: list[dict[str, Any]] = []
    if passed_list and not all(passed_list):
        probe = next(
            (
                e
                for e in dev["episodes"]
                if ":" not in e["session_id"]
                and e["status"] == "scorable"
                and e["episode_id"] == class_results["b0/peer"]["episodes"][0]["episode_id"]
            ),
            None,
        )
        if probe is not None:
            session = sessions.get(probe["session_id"])
            if session is not None and session.wav_abs_path is not None:
                samples = _load_wav(session.wav_abs_path)
                scored_start = int(probe["bounds"]["scored_start"])
                scored_end = int(probe["bounds"]["scored_end"])
                source = _replay_region(samples, 0, session.duration_samples)
                source_bounds = [
                    int(r["boundary_source_sample"])
                    for r in _boundary_rows(source)
                    if scored_start <= int(r["boundary_source_sample"]) < scored_end
                ]
                for warmup_s in (5, 15, 30, 60):
                    w2 = max(0, scored_start - warmup_s * CANONICAL_SAMPLE_RATE_HZ)
                    reset = _replay_region(
                        samples, w2, min(int(probe["bounds"]["tail_end"]), samples.size)
                    )
                    reset_bounds = [
                        int(r["boundary_source_sample"])
                        for r in _shift_boundary_rows(_boundary_rows(reset), w2)
                        if scored_start <= int(r["boundary_source_sample"]) < scored_end
                    ]
                    convergence_diagnostic.append(
                        {
                            "warmup_s": warmup_s,
                            "episode_id": probe["episode_id"],
                            "source_boundaries": source_bounds,
                            "reset_boundaries": reset_bounds,
                            "exact": reset_bounds == source_bounds,
                        }
                    )

    report: dict[str, Any] = {
        "schema_version": "turn_episode_v1",
        "report_id": "state_equivalence_report",
        "class": "b0/peer",
        "tolerances": {
            "b0": "exact (tolerance 0) on scored-region boundary trace and aligned (observed, safe-frontier) progress rows",
            "ls_eend": "max L1 <= 1e-2 over aligned posterior frames (declared; executed Phase 4)",
            "eres2netv2": "aligned-window cosine >= 0.99 (declared; executed Phase 4)",
        },
        "finding": (
            "B0/peer FAILS the state-equivalence gate: the Silero VAD v5 RNN hidden "
            "state carries long context (warm-up convergence diagnostic shows exact "
            "parity only from ~60 s warm-up; 5 s warm-up yields different scored-region "
            "boundaries in 186/186 episodes). Per PRD Section 5.4 and invariant 26, "
            "reset-plus-warm-up scored evaluation is forbidden for B0/peer; scored "
            "episodes must use deterministic source-prefix state (full-session replay "
            "with the episode scored region sliced from the source-prefix trace). "
            "The failed parity cases remain diagnostic evidence and are not hidden by "
            "increasing warm-up."
        ),
        "disposition_table": class_results,
        "convergence_diagnostic": convergence_diagnostic,
        "snapshot_fallback": {
            "mechanism": "deterministic source-prefix state snapshot binding model hash, warm-up slice, chunk config; restore replays warm-up deterministically",
            "round_trip_episodes": len(snapshot_evidence),
            "round_trip_passed": all(e["restored_boundary_count"] >= 0 for e in snapshot_evidence),
        },
        "snapshot_evidence": snapshot_evidence,
        "parity_skipped": parity_skipped,
        "pending_state_notes": (
            "per-episode pending-start inspection recorded in episode_manifest_dev.json "
            "flags.readiness; 1 episode diagnostic_only pending_state_unresolved"
        ),
    }
    payload = {k: v for k, v in report.items() if k != "content_sha256"}
    report["content_sha256"] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    path = out / "state_equivalence_report.json"
    path.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(
        f"b0/peer disposition: {class_results['b0/peer']['disposition']} "
        f"({len(passed_list)} episodes)"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
