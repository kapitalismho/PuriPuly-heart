"""Manifest row -> waveform bytes (issue #117, Gate 2 scaffold).

Audio lives outside the repo and is NEVER copied into it. Resolution order:

1. ``audio_ref``: taken from the row itself when present, else from the cached
   V2 GT interval sources (relative-occupancy manifests, dev + eval), else
   derived from the per-corpus convention
   (``ami/audio/<SID>/<SID>.Mix-Headset.wav``,
   ``alimeeting/far_ch0/<SID>.wav``).
2. Root: ``PSEM_CORPUS_ROOT`` first, ``PSEM_REFERENCE_ROOT`` as fallback.

All failures are fail-closed with the missing env var / path named.
"""

from __future__ import annotations

import hashlib
import json
import os
import wave
from pathlib import Path

SAMPLE_RATE_HZ = 16000
SAMPLES_PER_MS = SAMPLE_RATE_HZ // 1000  # 16, exact

REPO = Path(__file__).resolve().parents[3]
OCC_MANIFESTS = (
    REPO / "experiments/psem_relative_occupancy_gate/results/dev/relative_occupancy_manifest.jsonl",
    REPO / "experiments/psem_relative_occupancy_gate/results/eval/relative_occupancy_manifest.jsonl",
)

_AUDIO_REF_INDEX: dict[tuple[str, str], str] | None = None


def _convention_audio_ref(corpus: str, session_id: str) -> str:
    name = corpus.lower()
    if name == "ami":
        return f"ami/audio/{session_id}/{session_id}.Mix-Headset.wav"
    if name == "alimeeting":
        return f"alimeeting/far_ch0/{session_id}.wav"
    raise KeyError(
        f"no audio_ref convention for corpus={corpus!r} session={session_id!r}"
    )


def audio_ref_index() -> dict[tuple[str, str], str]:
    """(corpus.lower(), session_id) -> audio_ref from V2 GT sources."""
    global _AUDIO_REF_INDEX
    if _AUDIO_REF_INDEX is not None:
        return _AUDIO_REF_INDEX
    index: dict[tuple[str, str], str] = {}
    for path in OCC_MANIFESTS:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ref = row.get("audio_ref")
            if ref:
                index[(str(row["corpus"]).lower(), row["session_id"])] = ref
    _AUDIO_REF_INDEX = index
    return index


def audio_ref_for_row(row: dict) -> str:
    """Best-known audio_ref for a manifest row (never touches the filesystem)."""
    if row.get("audio_ref"):
        return str(row["audio_ref"])
    key = (str(row["corpus"]).lower(), row["session_id"])
    hit = audio_ref_index().get(key)
    if hit:
        return hit
    return _convention_audio_ref(str(row["corpus"]), row["session_id"])


def _candidate_roots() -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    for var in ("PSEM_CORPUS_ROOT", "PSEM_REFERENCE_ROOT"):
        raw = os.environ.get(var)
        if raw:
            roots.append((var, Path(raw)))
    return roots


def resolve_audio(row: dict) -> Path:
    """Resolve a manifest row to an on-disk WAV path (no copying).

    Raises FileNotFoundError naming the missing env var(s) or the tried paths.
    """
    ref = audio_ref_for_row(row)
    roots = _candidate_roots()
    if not roots:
        raise FileNotFoundError(
            "audio resolution needs PSEM_CORPUS_ROOT (primary) or "
            f"PSEM_REFERENCE_ROOT (fallback); neither is set (audio_ref={ref})"
        )
    tried = []
    for var, root in roots:
        candidate = root / ref
        if candidate.is_file():
            return candidate
        tried.append(f"{var}={root} -> {candidate} (missing)")
    raise FileNotFoundError(
        f"audio not found for corpus={row.get('corpus')} "
        f"session={row.get('session_id')} audio_ref={ref}; tried: "
        + "; ".join(tried)
    )


def load_span(path: Path, start_ms: int, end_ms: int) -> bytes:
    """Load [start_ms, end_ms) as mono 16 kHz int16-LE PCM bytes.

    Sample math is exact (16 samples/ms). Fail-closed: wrong WAV format,
    negative/empty span, or end past EOF all raise.
    """
    if end_ms <= start_ms or start_ms < 0:
        raise ValueError(
            f"empty/negative span: start_ms={start_ms} end_ms={end_ms}"
        )
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getframerate() != SAMPLE_RATE_HZ
            or reader.getnchannels() != 1
            or reader.getsampwidth() != 2
        ):
            raise ValueError(
                f"expected mono 16 kHz PCM16 WAV: {path} (got "
                f"{reader.getframerate()} Hz, {reader.getnchannels()} ch, "
                f"{reader.getsampwidth() * 8}-bit)"
            )
        total = reader.getnframes()
        start_sample = start_ms * SAMPLES_PER_MS
        end_sample = end_ms * SAMPLES_PER_MS
        if end_sample > total:
            raise ValueError(
                f"span [{start_ms},{end_ms})ms = samples "
                f"[{start_sample},{end_sample}) exceeds {total} frames in {path}"
            )
        reader.setpos(start_sample)
        return reader.readframes(end_sample - start_sample)


def sha256_pcm(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def span_for_regime(row: dict, regime: str) -> tuple[int, int]:
    """(start_ms, end_ms) bind span for regime 'O' (native 5 s) or 'C' (causal 1 s)."""
    if regime == "O":
        return int(row["native_reference_start_ms"]), int(row["native_reference_end_ms"])
    if regime == "C":
        start, end = row.get("causal_reference_start_ms"), row.get("causal_reference_end_ms")
        if start is None or end is None:
            raise ValueError(
                f"episode {row.get('episode_id')}: no causal span "
                f"(causal_bindable={row.get('causal_bindable')})"
            )
        return int(start), int(end)
    raise ValueError(f"unknown regime {regime!r} (want 'O' or 'C')")
