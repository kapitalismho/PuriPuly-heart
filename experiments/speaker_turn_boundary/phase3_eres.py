from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.speaker_turn_boundary.adapters.eres2netv2 import (
    ERES_SAMPLE_RATE_HZ,
    EresAdjacentProfile,
    EresEmbeddingRuntime,
    EresStableAnchorProfile,
    clamp_confidence,
    cosine_similarity,
)
from experiments.speaker_turn_boundary.events import DetectorProgress, SpeakerBoundaryEvent
from experiments.speaker_turn_boundary.schemas import canonical_json, sha256_hex

ERES_CACHE_SCHEMA = "experiments.speaker_turn_boundary.phase3.eres_cache.v2"


class EresCacheError(RuntimeError):
    pass


def _sample_count(seconds: float) -> int:
    return int(round(seconds * ERES_SAMPLE_RATE_HZ))


def sanitize_case_id(case_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in case_id)


def enumerate_adjacent_windows(
    utterances: list[tuple[int, int]],
    grid: list[tuple[float, float]],
) -> set[tuple[int, int]]:
    windows: set[tuple[int, int]] = set()
    for start, end in utterances:
        for window_seconds, step_seconds in grid:
            window = _sample_count(window_seconds)
            step = _sample_count(step_seconds)
            position = start + window
            while position + window <= end:
                windows.add((position - window, position))
                windows.add((position, position + window))
                position += step
    return windows


def enumerate_anchor_windows(
    utterances: list[tuple[int, int]],
    grid: list[tuple[float, float]],
) -> set[tuple[int, int]]:
    windows: set[tuple[int, int]] = set()
    for start, end in utterances:
        for window_seconds, step_seconds in grid:
            window = _sample_count(window_seconds)
            step = _sample_count(step_seconds)
            if start + window > end:
                continue
            windows.add((start, start + window))
            position = start + window
            while position + window <= end:
                windows.add((position, position + window))
                position += step
    return windows


def _window_key(window: tuple[int, int]) -> str:
    return f"{window[0]}-{window[1]}"


def _decode_window_key(key: str) -> tuple[int, int]:
    start, end = key.split("-", maxsplit=1)
    return int(start), int(end)


@dataclass(slots=True)
class EresEmbeddingStore:
    checkpoint_tag: str
    checkpoint_sha256: str
    frontend_contract_sha256: str
    manifest_sha256: str
    cache_dir: Path
    window_count_computed: int = 0
    window_count_loaded: int = 0
    embed_seconds: list[float] = field(default_factory=list)

    def _base_path(self, case_id: str, wav_sha256: str) -> Path:
        name = f"{sanitize_case_id(case_id)}_{wav_sha256[:16]}"
        return self.cache_dir / self.checkpoint_tag / self.manifest_sha256[:16] / name

    def load_case(
        self,
        case_id: str,
        wav_sha256: str,
    ) -> dict[tuple[int, int], np.ndarray]:
        base = self._base_path(case_id, wav_sha256)
        npz_path = base.with_suffix(".npz")
        metadata_path = base.with_suffix(".json")
        if not npz_path.is_file() or not metadata_path.is_file():
            return {}
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": ERES_CACHE_SCHEMA,
            "checkpoint": self.checkpoint_tag,
            "checkpoint_sha256": self.checkpoint_sha256,
            "frontend_contract_sha256": self.frontend_contract_sha256,
            "manifest_sha256": self.manifest_sha256,
            "case_id": case_id,
            "wav_sha256": wav_sha256,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise EresCacheError(f"cache contract mismatch for {case_id}: {key}")
        actual_file_hash = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        if metadata.get("npz_sha256") != actual_file_hash:
            raise EresCacheError(f"cache byte hash mismatch for {case_id}")
        embeddings: dict[tuple[int, int], np.ndarray] = {}
        with np.load(npz_path, allow_pickle=False) as data:
            for key in data.files:
                vector = np.asarray(data[key], dtype=np.float32).reshape(-1)
                if vector.size != 192 or not np.all(np.isfinite(vector)):
                    raise EresCacheError(f"invalid embedding array {case_id}:{key}")
                embeddings[_decode_window_key(key)] = vector
        if sorted(metadata.get("windows") or []) != sorted(
            data_key for data_key in map(_window_key, embeddings)
        ):
            raise EresCacheError(f"cache window index mismatch for {case_id}")
        self.window_count_loaded += len(embeddings)
        return embeddings

    def save_case(
        self,
        case_id: str,
        wav_sha256: str,
        embeddings: dict[tuple[int, int], np.ndarray],
        *,
        import_evidence: dict[str, Any] | None = None,
    ) -> None:
        base = self._base_path(case_id, wav_sha256)
        base.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            _window_key(window): np.asarray(vector, dtype=np.float32)
            for window, vector in sorted(embeddings.items())
        }
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        payload = buffer.getvalue()
        npz_path = base.with_suffix(".npz")
        npz_path.write_bytes(payload)
        metadata: dict[str, Any] = {
            "schema_version": ERES_CACHE_SCHEMA,
            "checkpoint": self.checkpoint_tag,
            "checkpoint_sha256": self.checkpoint_sha256,
            "frontend_contract_sha256": self.frontend_contract_sha256,
            "manifest_sha256": self.manifest_sha256,
            "case_id": case_id,
            "wav_sha256": wav_sha256,
            "windows": sorted(arrays),
            "npz_sha256": hashlib.sha256(payload).hexdigest(),
        }
        if import_evidence is not None:
            metadata["legacy_import_evidence"] = import_evidence
        metadata["contract_sha256"] = sha256_hex(metadata)
        base.with_suffix(".json").write_text(canonical_json(metadata), encoding="utf-8")

    def ensure_case(
        self,
        runtime: EresEmbeddingRuntime,
        samples: np.ndarray,
        case_id: str,
        wav_sha256: str,
        windows: set[tuple[int, int]],
        *,
        imported: dict[tuple[int, int], np.ndarray] | None = None,
        import_evidence: dict[str, Any] | None = None,
    ) -> dict[tuple[int, int], np.ndarray]:
        embeddings = self.load_case(case_id, wav_sha256)
        imported_used = False
        if not embeddings and imported:
            embeddings.update(imported)
            imported_used = True
        missing = sorted(window for window in windows if window not in embeddings)
        for start, end in missing:
            begin = time.perf_counter()
            vector = runtime.embed(samples[start:end])
            self.embed_seconds.append(time.perf_counter() - begin)
            vector = np.asarray(vector, dtype=np.float32).reshape(-1)
            if vector.size != 192 or not np.all(np.isfinite(vector)):
                raise EresCacheError(
                    f"runtime returned invalid embedding for {case_id}:{start}-{end}"
                )
            embeddings[(start, end)] = vector
        self.window_count_computed += len(missing)
        if missing or imported_used:
            self.save_case(
                case_id,
                wav_sha256,
                embeddings,
                import_evidence=import_evidence,
            )
        return {window: embeddings[window] for window in windows}


def load_legacy_case(path: Path) -> dict[tuple[int, int], np.ndarray]:
    if not path.is_file():
        return {}
    embeddings: dict[tuple[int, int], np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            vector = np.asarray(data[key], dtype=np.float32).reshape(-1)
            if vector.size != 192 or not np.all(np.isfinite(vector)):
                raise EresCacheError(f"legacy cache has invalid embedding {path}:{key}")
            embeddings[_decode_window_key(key)] = vector
    return embeddings


def verify_legacy_samples(
    runtime: EresEmbeddingRuntime,
    samples: np.ndarray,
    embeddings: dict[tuple[int, int], np.ndarray],
    windows: list[tuple[int, int]],
) -> dict[str, Any]:
    max_abs = 0.0
    min_cosine = 1.0
    for start, end in windows:
        expected = embeddings[(start, end)]
        actual = np.asarray(runtime.embed(samples[start:end]), dtype=np.float32).reshape(-1)
        max_abs = max(max_abs, float(np.max(np.abs(expected - actual))))
        min_cosine = min(min_cosine, cosine_similarity(expected, actual))
    passed = max_abs <= 1e-5 and min_cosine >= 0.99999
    evidence = {
        "sample_count": len(windows),
        "max_abs_error": max_abs,
        "min_cosine_similarity": min_cosine,
        "passed": passed,
    }
    if not passed:
        raise EresCacheError(f"legacy embedding verification failed: {evidence}")
    return evidence


def _change_confidence(scores: list[float]) -> float:
    return clamp_confidence(float(np.mean(scores)))


def _event(
    *,
    audio_epoch: int,
    boundary_sample: int,
    observed_sample: int,
    confidence: float,
    profile_id: str,
    debug: dict[str, Any],
) -> SpeakerBoundaryEvent:
    return SpeakerBoundaryEvent(
        audio_epoch=audio_epoch,
        boundary_source_sample=boundary_sample,
        observed_source_sample_at_emit=observed_sample,
        emitted_monotonic_ns=0,
        confidence=confidence,
        source=f"eres2netv2:{profile_id}",
        debug=debug,
    )


def cached_adjacent_events(
    *,
    utterances: list[tuple[int, int]],
    embeddings: dict[tuple[int, int], np.ndarray],
    profile: EresAdjacentProfile,
    audio_epoch: int,
) -> tuple[list[SpeakerBoundaryEvent], list[DetectorProgress]]:
    window = _sample_count(profile.window_seconds)
    step = _sample_count(profile.step_seconds)
    events: list[SpeakerBoundaryEvent] = []
    progress: list[DetectorProgress] = []
    for start_sample, end_sample in utterances:
        pending: tuple[float, int] | None = None
        latched = False
        position = start_sample + window
        while position + window <= end_sample:
            score = cosine_similarity(
                embeddings[(position - window, position)],
                embeddings[(position, position + window)],
            )
            candidate = score < profile.threshold
            if latched:
                if not candidate:
                    latched = False
                pending = None
            elif profile.confirmation == 1:
                if candidate:
                    events.append(
                        _event(
                            audio_epoch=audio_epoch,
                            boundary_sample=position,
                            observed_sample=position + window,
                            confidence=_change_confidence([score]),
                            profile_id=profile.profile_id,
                            debug={
                                "profile": profile.to_dict(),
                                "scores": [score],
                                "confirmation_positions": [position],
                            },
                        )
                    )
                    latched = True
                pending = None
            elif candidate and pending is None:
                pending = (score, position)
            elif candidate and pending is not None:
                first_score, first_position = pending
                events.append(
                    _event(
                        audio_epoch=audio_epoch,
                        boundary_sample=first_position,
                        observed_sample=position + window,
                        confidence=_change_confidence([first_score, score]),
                        profile_id=profile.profile_id,
                        debug={
                            "profile": profile.to_dict(),
                            "scores": [first_score, score],
                            "confirmation_positions": [first_position, position],
                        },
                    )
                )
                pending = None
                latched = True
            else:
                pending = None
            frontier = pending[1] if pending is not None else min(end_sample, position + step)
            progress.append(
                DetectorProgress(
                    audio_epoch=audio_epoch,
                    observed_source_sample=position + window,
                    safe_boundary_frontier_sample=frontier,
                )
            )
            position += step
        progress.append(
            DetectorProgress(
                audio_epoch=audio_epoch,
                observed_source_sample=end_sample,
                safe_boundary_frontier_sample=end_sample,
            )
        )
    return events, progress


def _normalized(vector: np.ndarray) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + 1e-12)


def cached_anchor_events(
    *,
    utterances: list[tuple[int, int]],
    embeddings: dict[tuple[int, int], np.ndarray],
    profile: EresStableAnchorProfile,
    audio_epoch: int,
) -> tuple[list[SpeakerBoundaryEvent], list[DetectorProgress]]:
    window = _sample_count(profile.window_seconds)
    step = _sample_count(profile.step_seconds)
    events: list[SpeakerBoundaryEvent] = []
    progress: list[DetectorProgress] = []
    for start_sample, end_sample in utterances:
        if start_sample + window > end_sample:
            continue
        anchor = _normalized(embeddings[(start_sample, start_sample + window)])
        pending: tuple[float, int, np.ndarray] | None = None
        position = start_sample + window
        while position + window <= end_sample:
            probe = embeddings[(position, position + window)]
            score = cosine_similarity(anchor, probe)
            candidate = score < profile.threshold
            if profile.confirmation == 1:
                if candidate:
                    events.append(
                        _event(
                            audio_epoch=audio_epoch,
                            boundary_sample=position,
                            observed_sample=position + window,
                            confidence=_change_confidence([score]),
                            profile_id=profile.profile_id,
                            debug={
                                "profile": profile.to_dict(),
                                "anchor_scores": [score],
                                "confirmation_positions": [position],
                                "mutual_similarity": None,
                            },
                        )
                    )
                    anchor = _normalized(probe)
                elif profile.anchor_update == "ema":
                    alpha = profile.anchor_ema_alpha
                    anchor = _normalized((1.0 - alpha) * anchor + alpha * _normalized(probe))
                pending = None
            elif candidate and pending is None:
                pending = (score, position, probe)
            elif candidate and pending is not None:
                first_score, first_position, first_probe = pending
                mutual = cosine_similarity(first_probe, probe)
                if mutual >= profile.mutual_similarity_threshold:
                    events.append(
                        _event(
                            audio_epoch=audio_epoch,
                            boundary_sample=first_position,
                            observed_sample=position + window,
                            confidence=_change_confidence([first_score, score]),
                            profile_id=profile.profile_id,
                            debug={
                                "profile": profile.to_dict(),
                                "anchor_scores": [first_score, score],
                                "confirmation_positions": [first_position, position],
                                "mutual_similarity": mutual,
                            },
                        )
                    )
                    anchor = _normalized(first_probe)
                    pending = None
                else:
                    pending = (score, position, probe)
            else:
                pending = None
                if profile.anchor_update == "ema":
                    alpha = profile.anchor_ema_alpha
                    anchor = _normalized((1.0 - alpha) * anchor + alpha * _normalized(probe))
            frontier = pending[1] if pending is not None else min(end_sample, position + step)
            progress.append(
                DetectorProgress(
                    audio_epoch=audio_epoch,
                    observed_source_sample=position + window,
                    safe_boundary_frontier_sample=frontier,
                )
            )
            position += step
        progress.append(
            DetectorProgress(
                audio_epoch=audio_epoch,
                observed_source_sample=end_sample,
                safe_boundary_frontier_sample=end_sample,
            )
        )
    return events, progress
