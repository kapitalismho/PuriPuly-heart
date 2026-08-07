from __future__ import annotations

import wave

import numpy as np
import pytest

from experiments.speaker_turn_boundary.events import DetectorProgress
from experiments.speaker_turn_boundary.tests.helpers import (
    SequenceVadEngine,
    chunk_samples,
    write_pcm16_wav,
)
from experiments.speaker_turn_boundary.vad_baseline import (
    CanonicalAudioError,
    VadBoundaryReplay,
    load_canonical_wav,
    replay_wav_epoch,
)


def scripted(probs: list[float]) -> VadBoundaryReplay:
    replay = VadBoundaryReplay(
        engine_factory=lambda: SequenceVadEngine(probs=probs),
        hangover_ms=64,
        monotonic_ns=lambda: 1000,
    )
    return replay


def feed(replay: VadBoundaryReplay, probs: list[float]) -> list[object]:
    replay.start_epoch(0)
    boundaries = []
    for value in probs:
        boundaries.extend(replay.process_chunk(chunk_samples(value)))
    return boundaries


def test_first_utterance_emits_no_boundary_and_second_utterance_emits_one():
    probs = [0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.4, 0.4, 0.0, 0.9, 0.9, 0.9, 0.4, 0.4, 0.0]
    boundaries = feed(scripted(probs), probs)
    assert len(boundaries) == 1
    boundary = boundaries[0]
    assert boundary.audio_epoch == 0
    assert boundary.boundary_source_sample == 11 * 512
    assert boundary.observed_source_sample_at_emit == 12 * 512
    assert boundary.source == "vad_b0"
    assert boundary.confidence is None
    assert boundary.debug["pre_roll_samples"] == 9 * 512


def test_boundary_debug_reports_previous_utterance_end_and_gap():
    probs = [0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.4, 0.4, 0.0, 0.9, 0.9, 0.9, 0.4, 0.4, 0.0]
    boundary = feed(scripted(probs), probs)[0]
    debug = boundary.debug
    assert debug["prev_speech_end_sample"] == 6 * 512
    assert debug["gap_samples"] == 11 * 512 - 6 * 512
    assert debug["prev_trailing_silence_ms"] == 64
    assert debug["prev_end_reason"] == "silence"
    assert debug["start_chunk_index"] == 11
    assert debug["chunk_samples"] == 512


def test_no_boundary_when_stream_ends_without_next_utterance():
    probs = [0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.4, 0.4, 0.0]
    boundaries = feed(scripted(probs), probs)
    assert boundaries == []


def test_max_duration_end_still_derives_speech_end_sample():
    probs = [0.9] * 221 + [0.0, 0.0]
    replay = VadBoundaryReplay(
        engine_factory=lambda: SequenceVadEngine(probs=probs),
        hangover_ms=64,
        monotonic_ns=lambda: 1000,
    )
    boundaries = feed(replay, probs)
    assert len(boundaries) == 0
    assert replay.progress_snapshot().safe_boundary_frontier_sample >= 0


def test_max_duration_utterance_then_next_utterance_boundary_reason():
    probs = [0.9] * 221 + [0.0, 0.0, 0.9, 0.9, 0.9, 0.4, 0.4, 0.0]
    replay = VadBoundaryReplay(
        engine_factory=lambda: SequenceVadEngine(probs=probs),
        hangover_ms=64,
        monotonic_ns=lambda: 1000,
    )
    boundaries = feed(replay, probs)
    assert len(boundaries) == 1
    boundary = boundaries[0]
    assert boundary.boundary_source_sample == 225 * 512
    assert boundary.debug["prev_end_reason"] == "max_duration"
    assert boundary.debug["prev_trailing_silence_ms"] == 0
    assert boundary.debug["prev_speech_end_sample"] == 219 * 512


def test_replay_is_deterministic_with_injected_clock():
    probs = [0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.4, 0.4, 0.0, 0.9, 0.9, 0.9, 0.4, 0.4, 0.0]
    runs = []
    for run_index in range(2):
        clock_values = iter(range(5000, 6000))
        replay = VadBoundaryReplay(
            engine_factory=lambda: SequenceVadEngine(probs=probs),
            hangover_ms=64,
            monotonic_ns=lambda: next(clock_values),
        )
        boundaries = feed(replay, probs)
        runs.append([boundary.to_dict() for boundary in boundaries])
    assert runs[0] == runs[1]


def test_progress_invariants_and_monotonicity():
    probs = [0.0, 0.0, 0.9, 0.9, 0.9, 0.4, 0.4, 0.0]
    replay = scripted(probs)
    feed(replay, probs)
    progress = replay.progress
    assert len(progress) == len(probs)
    for index, snapshot in enumerate(progress):
        assert snapshot.audio_epoch == 0
        assert snapshot.observed_source_sample == (index + 1) * 512
        assert snapshot.safe_boundary_frontier_sample == index * 512
    observed = [snapshot.observed_source_sample for snapshot in progress]
    safe = [snapshot.safe_boundary_frontier_sample for snapshot in progress]
    assert observed == sorted(observed)
    assert safe == sorted(safe)
    assert all(
        snapshot.safe_boundary_frontier_sample <= snapshot.observed_source_sample
        for snapshot in progress
    )


def test_progress_snapshot_requires_processed_epoch():
    replay = scripted([0.0])
    with pytest.raises(RuntimeError):
        replay.progress_snapshot()


def test_process_chunk_requires_start_epoch():
    replay = scripted([0.0])
    with pytest.raises(RuntimeError):
        replay.process_chunk(chunk_samples(0.0))


def test_process_chunk_requires_exact_chunk_size():
    replay = scripted([0.0])
    replay.start_epoch(0)
    with pytest.raises(ValueError):
        replay.process_chunk(np.zeros(511, dtype=np.float32))


def test_start_epoch_resets_state_and_uses_fresh_engine():
    probs = [0.0, 0.0, 0.9, 0.9, 0.9, 0.4, 0.4, 0.0]
    replay = scripted(probs)
    feed(replay, probs)
    replay.start_epoch(1)
    boundaries = []
    for value in probs:
        boundaries.extend(replay.process_chunk(chunk_samples(value)))
    assert all(boundary.audio_epoch == 1 for boundary in boundaries)
    assert replay.audio_epoch == 1


def test_load_canonical_wav_round_trip(tmp_dir):
    samples = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    wav_path = tmp_dir / "roundtrip.wav"
    write_pcm16_wav(wav_path, samples)
    loaded = load_canonical_wav(wav_path)
    assert loaded.size == 1000
    assert np.allclose(loaded, samples, atol=1.0 / 32767.0)


def test_load_canonical_wav_rejects_wrong_sample_rate(tmp_dir):
    samples = np.zeros(1600, dtype=np.float32)
    wav_path = tmp_dir / "wrong_rate.wav"
    write_pcm16_wav(wav_path, samples, sample_rate_hz=8000)
    with pytest.raises(CanonicalAudioError):
        load_canonical_wav(wav_path)


def test_load_canonical_wav_rejects_stereo(tmp_dir):
    wav_path = tmp_dir / "stereo.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(np.zeros(1600, dtype="<i2").tobytes())
    with pytest.raises(CanonicalAudioError):
        load_canonical_wav(wav_path)


def test_load_canonical_wav_rejects_8bit(tmp_dir):
    wav_path = tmp_dir / "eight_bit.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(1)
        wav_file.setframerate(16000)
        wav_file.writeframes(np.zeros(1600, dtype="<i1").tobytes())
    with pytest.raises(CanonicalAudioError):
        load_canonical_wav(wav_path)


def test_replay_wav_epoch_end_to_end(tmp_dir):
    probs = [0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.4, 0.4, 0.0, 0.9, 0.9, 0.9, 0.4, 0.4, 0.0]
    clip = np.concatenate([chunk_samples(value) for value in probs])
    wav_path = tmp_dir / "clip.wav"
    write_pcm16_wav(wav_path, clip)
    result = replay_wav_epoch(
        wav_path,
        audio_epoch=3,
        engine_factory=lambda: SequenceVadEngine(probs=probs),
        monotonic_ns=lambda: 42,
        hangover_ms=64,
    )
    assert result.audio_epoch == 3
    assert result.length_samples == clip.size
    assert len(result.boundaries) == 1
    assert result.boundaries[0].boundary_source_sample == 11 * 512
    assert len(result.progress) == 15
    assert result.progress[-1].observed_source_sample == 15 * 512
    assert all(isinstance(snapshot, DetectorProgress) for snapshot in result.progress)


def test_replay_wav_epoch_ignores_partial_trailing_chunk(tmp_dir):
    clip = np.concatenate([chunk_samples(0.0), np.zeros(100, dtype=np.float32)])
    wav_path = tmp_dir / "partial.wav"
    write_pcm16_wav(wav_path, clip)
    result = replay_wav_epoch(
        wav_path,
        audio_epoch=0,
        engine_factory=lambda: SequenceVadEngine(probs=[0.0, 0.0]),
        monotonic_ns=lambda: 0,
    )
    assert result.length_samples == clip.size
    assert len(result.progress) == 1
