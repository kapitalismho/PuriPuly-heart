from __future__ import annotations

import wave

import numpy as np
import pytest

from experiments.psem_ontology_simplification_gate import run_production_vad
from experiments.psem_ontology_simplification_gate.run_production_vad import (
    ProductionVadReplayError,
    _merge_spans,
    _replay_source,
    _validate_profile,
)
from experiments.psem_relative_occupancy_gate.io_utils import sha256_file


def test_production_vad_spans_merge_touching_and_overlapping_support() -> None:
    assert _merge_spans([(20, 30), (0, 10), (10, 25), (40, 40)]) == [(0, 30)]


def test_production_vad_profile_binds_runtime_constants() -> None:
    profile = {
        "profile": "peer",
        "backend": "CPUExecutionProvider",
        "start_debounce_chunks": 3,
        "start_commit_chunks": 3,
        "max_segment_ms": 7000,
        "chunk_samples": 512,
        "source_support": "pre_roll_plus_committed_chunks_through_speech_end_excluding_trailing_hangover",
    }
    _validate_profile(profile)
    profile["start_commit_chunks"] = 2
    with pytest.raises(ProductionVadReplayError, match="start_commit_chunks"):
        _validate_profile(profile)


def test_production_vad_replay_binds_audio_and_processes_partial_tail(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "source.wav"
    samples = np.zeros(513, dtype="<i2")
    with wave.open(str(audio_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(samples.tobytes())

    class Gating:
        def process_chunk(self, chunk):
            assert len(chunk) == 512
            return []

    monkeypatch.setattr(run_production_vad, "SileroVadOnnx", lambda path: object())
    monkeypatch.setattr(run_production_vad, "create_peer_vad_gating", lambda *args, **kwargs: Gating())
    row = {
        "source_id": "source",
        "audio_path": str(audio_path),
        "waveform_sha256": sha256_file(audio_path),
        "waveform_size_bytes": audio_path.stat().st_size,
        "source_duration_samples": 513,
        "scored_start_sample": 0,
        "scored_end_sample": 513,
    }
    cfg = {
        "speech_gate": {
            "production_vad": {
                "pre_roll_ms": 500,
                "speech_threshold": 0.5,
                "hangover_ms": 500,
                "chunk_samples": 512,
            }
        }
    }
    result = _replay_source(row, cfg)
    assert result["audio_size_bytes"] == audio_path.stat().st_size
    assert result["processed_samples"] == 513
    assert result["ignored_tail_samples"] == 0
    row["waveform_sha256"] = "0" * 64
    with pytest.raises(ProductionVadReplayError, match="audio digest mismatch"):
        _replay_source(row, cfg)
