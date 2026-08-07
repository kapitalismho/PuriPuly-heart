from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.speaker_turn_boundary.build_synthetic_cases import build as build_cases
from experiments.speaker_turn_boundary.schemas import (
    DatasetManifest,
    validate_manifest,
)
from experiments.speaker_turn_boundary.synthetic import (
    build_default_cases,
    pcm16_bytes,
)
from experiments.speaker_turn_boundary.tests.helpers import write_pcm16_wav
from experiments.speaker_turn_boundary.vad_baseline import (
    replay_wav_epoch,
)
from puripuly_heart.core.vad.gating import create_peer_vad_gating
from puripuly_heart.core.vad.silero import SileroVadOnnx

CHUNK = 512


def _clip_audio(case_id: str) -> np.ndarray:
    cases = {case.case_id: case for case in build_default_cases()}
    return np.asarray(cases[case_id].audio, dtype=np.float32)


def _raw_dev_trace(clip: np.ndarray, model_path: Path):
    engine = SileroVadOnnx(model_path)
    gating = create_peer_vad_gating(
        engine,
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        hangover_ms=500,
    )
    events = []
    for chunk_index in range(clip.size // CHUNK):
        chunk = clip[chunk_index * CHUNK : (chunk_index + 1) * CHUNK]
        for event in gating.process_chunk(chunk):
            events.append((chunk_index, event))
    return events


def _derive_expected_boundaries(raw_events):
    boundaries = []
    current = None
    previous = None
    for chunk_index, event in raw_events:
        kind = type(event).__name__
        if kind == "SpeechStart":
            if previous is not None:
                boundaries.append(
                    {
                        "boundary_source_sample": chunk_index * CHUNK,
                        "observed_source_sample_at_emit": (chunk_index + 1) * CHUNK,
                        "prev_speech_end_sample": previous["speech_end_sample"],
                        "gap_samples": chunk_index * CHUNK - previous["speech_end_sample"],
                        "prev_trailing_silence_ms": previous["trailing_silence_ms"],
                        "prev_end_reason": previous["reason"],
                    }
                )
            current = {"start": chunk_index * CHUNK}
        elif kind == "SpeechEnd":
            if current is None:
                continue
            trailing_silence_ms = int(getattr(event, "trailing_silence_ms", 0))
            reason = str(getattr(event, "reason", "silence"))
            silence_run = int(round(trailing_silence_ms / (CHUNK / 16000.0 * 1000.0)))
            previous = {
                "speech_end_sample": (chunk_index + 1 - silence_run) * CHUNK,
                "trailing_silence_ms": trailing_silence_ms,
                "reason": reason,
            }
            current = None
    return boundaries


def _silero_factory(model_path: Path):
    return lambda: SileroVadOnnx(model_path)


def _run_replay(clip: np.ndarray, model_path: Path, tmp_dir: Path, clock):
    wav_path = tmp_dir / "clip.wav"
    write_pcm16_wav(wav_path, clip)
    result = replay_wav_epoch(
        wav_path,
        audio_epoch=0,
        engine_factory=_silero_factory(model_path),
        monotonic_ns=clock,
    )
    return result


def test_real_model_replay_is_deterministic(silero_model_path, tmp_dir):
    clip = _clip_audio("golden_two_utterance_gap400")
    runs = []
    for _ in range(2):
        clock = iter(range(100000, 100000 + 500))
        result = _run_replay(clip, silero_model_path, tmp_dir, lambda: next(clock))
        runs.append(
            {
                "boundaries": [b.to_dict() for b in result.boundaries],
                "progress": [p.to_dict() for p in result.progress],
                "length_samples": result.length_samples,
            }
        )
    assert runs[0] == runs[1]


def test_real_model_replay_equivalence_with_raw_dev_pipeline(silero_model_path, tmp_dir):
    clip = _clip_audio("golden_two_utterance_gap400")
    raw_events = _raw_dev_trace(clip, silero_model_path)
    expected = _derive_expected_boundaries(raw_events)
    clock = iter(range(100000, 100000 + 500))
    result = _run_replay(clip, silero_model_path, tmp_dir, lambda: next(clock))
    assert len(result.boundaries) == len(expected)
    for boundary, expected_boundary in zip(result.boundaries, expected, strict=True):
        assert boundary.audio_epoch == 0
        assert boundary.source == "vad_b0"
        assert boundary.boundary_source_sample == expected_boundary["boundary_source_sample"]
        assert (
            boundary.observed_source_sample_at_emit
            == expected_boundary["observed_source_sample_at_emit"]
        )
        assert (
            boundary.debug["prev_speech_end_sample"] == expected_boundary["prev_speech_end_sample"]
        )
        assert boundary.debug["gap_samples"] == expected_boundary["gap_samples"]
        assert (
            boundary.debug["prev_trailing_silence_ms"]
            == expected_boundary["prev_trailing_silence_ms"]
        )
        assert boundary.debug["prev_end_reason"] == expected_boundary["prev_end_reason"]


def test_real_model_replay_progress_invariants(silero_model_path, tmp_dir):
    clip = _clip_audio("golden_silence")
    result = _run_replay(clip, silero_model_path, tmp_dir, lambda: 1)
    assert result.boundaries == []
    assert result.length_samples == clip.size
    for snapshot in result.progress:
        assert snapshot.safe_boundary_frontier_sample == (snapshot.observed_source_sample - CHUNK)
    assert result.progress[-1].observed_source_sample == (clip.size // CHUNK) * CHUNK


def test_real_model_replay_silence_clip_emits_no_boundaries(silero_model_path, tmp_dir):
    clip = _clip_audio("golden_silence")
    result = _run_replay(clip, silero_model_path, tmp_dir, lambda: 1)
    assert result.boundaries == []


def test_committed_manifest_validates_against_regenerated_wavs(tmp_dir):
    committed_path = Path(__file__).resolve().parents[1] / "data" / "manifests" / "b0_phase0.json"
    if not committed_path.is_file():
        pytest.skip("committed b0_phase0 manifest not present")
    build_dir = tmp_dir / "data"
    build_cases(build_dir, seed=7, manifest_id="b0_phase0")
    committed = DatasetManifest.load(committed_path)
    validate_manifest(committed, build_dir)


def test_regenerated_audio_matches_committed_manifest_hashes(tmp_dir):
    import hashlib

    committed_path = Path(__file__).resolve().parents[1] / "data" / "manifests" / "b0_phase0.json"
    if not committed_path.is_file():
        pytest.skip("committed b0_phase0 manifest not present")
    build_dir = tmp_dir / "data"
    build_cases(build_dir, seed=7, manifest_id="b0_phase0")
    committed = DatasetManifest.load(committed_path)
    by_id = {case.case_id: case for case in committed.cases}
    generated = {case.case_id: case for case in build_default_cases()}
    assert by_id.keys() == generated.keys()
    for case_id, manifest_case in by_id.items():
        wav_path = build_dir / "generated" / f"{case_id}.wav"
        pcm = pcm16_bytes(np.asarray(generated[case_id].audio, dtype=np.float32))
        assert manifest_case.duration_samples == len(pcm) // 2
        assert manifest_case.wav_sha256 == hashlib.sha256(wav_path.read_bytes()).hexdigest()
