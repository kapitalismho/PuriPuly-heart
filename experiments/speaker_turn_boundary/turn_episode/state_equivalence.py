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
    evidence: dict[str, Any] = {
        "frontier_at_scored_start": _progress_rows(replay)[-1] if replay.progress else None,
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
        "frontier_at_final_scored_start": _progress_rows(replay)[-1] if replay.progress else None,
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
    processed_end = min(bounds.scored_end, effective_end - effective_end % CHUNK_SAMPLES)
    scored_interval = (bounds.scored_start, processed_end)

    def in_scored(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if scored_interval[0] <= int(row["boundary_source_sample"]) < scored_interval[1]
        ]

    source_bounds = in_scored(_boundary_rows(source))
    reset_bounds = in_scored(_boundary_rows(reset))
    source_progress = [
        row
        for row in _progress_rows(source)
        if scored_interval[0] <= int(row["observed_source_sample"]) <= scored_interval[1]
    ]
    reset_progress = [
        row
        for row in _progress_rows(reset)
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
    equal = (
        source_bounds == reset_bounds
        and source_progress == reset_progress
        and len(_boundary_rows(source)) == len(_boundary_rows(reset))
    )
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
