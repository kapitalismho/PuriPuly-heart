from __future__ import annotations

import numpy as np
import pytest

from experiments.speaker_turn_boundary.frontend import (
    LS_EEND_FFT_SIZE,
    LS_EEND_FRAME_HOP_8K,
    LS_EEND_LEFT_PAD_8K,
    LS_EEND_MODEL_INPUT_DIM,
    LS_EEND_SUBSAMPLING,
    RESAMPLER_CENTER_16K,
    RESAMPLER_TAPS,
    Resampler16k8k,
    StreamingLSEENDFrontend,
    available_8k_count,
    extract_logmel23_cummn_offline,
    model_frame_count_offline,
    model_input_frame_center_8k,
    model_input_frame_required_16k_count,
    output_frame_available_16k_count,
    output_frame_center_16k,
    output_frame_lookback_16k,
    stft_frame_count_offline,
    stream_whole_file,
)


def test_resampler_determinism():
    rng = np.random.default_rng(11)
    samples = rng.normal(0, 0.1, 20000).astype(np.float32)
    first = Resampler16k8k().push(samples)
    second = Resampler16k8k().push(samples)
    assert np.array_equal(first, second)


def test_resampler_chunked_equals_whole():
    rng = np.random.default_rng(23)
    samples = rng.normal(0, 0.1, 32137).astype(np.float32)
    whole = Resampler16k8k().push(samples)
    chunked = Resampler16k8k()
    parts = []
    offset = 0
    while offset < samples.size:
        parts.append(chunked.push(samples[offset : offset + 512]))
        offset += 512
    combined = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    assert combined.size == whole.size
    assert np.array_equal(combined, whole)


def test_resampler_output_count_formula():
    rng = np.random.default_rng(7)
    for count in (512, 1000, 80022, 15999):
        samples = rng.normal(0, 0.1, count).astype(np.float32)
        emitted = Resampler16k8k().push(samples).size
        expected = max(0, (count - RESAMPLER_CENTER_16K + 1) // 2)
        assert emitted == expected


def test_available_8k_count_matches_resampler():
    resampler = Resampler16k8k()
    assert available_8k_count(resampler.input_count) == 0
    rng = np.random.default_rng(5)
    for _ in range(20):
        resampler.push(rng.normal(0, 0.1, 512).astype(np.float32))
        assert resampler.emitted_count == available_8k_count(resampler.input_count)


def test_pinned_frame_mapping_constants():
    assert LS_EEND_FFT_SIZE == 256
    assert LS_EEND_LEFT_PAD_8K == 128
    assert LS_EEND_MODEL_INPUT_DIM == 345
    assert model_input_frame_center_8k(0) == 0
    assert model_input_frame_required_16k_count(0) == 1406
    assert output_frame_center_16k(0) == 14431
    assert output_frame_available_16k_count(0) == 15806
    assert output_frame_lookback_16k() == 1375
    assert output_frame_center_16k(9) == 28831
    assert output_frame_available_16k_count(9) == 30206


def test_mapping_center_inside_available():
    for frame in range(50):
        assert output_frame_center_16k(frame) < output_frame_available_16k_count(frame)


def test_stft_frame_counts():
    assert stft_frame_count_offline(0) == 0
    assert stft_frame_count_offline(80) == 1
    assert stft_frame_count_offline(160) == 2
    assert stft_frame_count_offline(16000) == 200
    assert model_frame_count_offline(16000) == 20
    assert model_frame_count_offline(801) == 2


def test_offline_features_shape_and_content():
    rng = np.random.default_rng(3)
    audio = rng.normal(0, 0.1, 16000).astype(np.float32)
    features = extract_logmel23_cummn_offline(audio)
    assert features.shape[0] == model_frame_count_offline(audio.size)
    assert features.shape[1] == LS_EEND_MODEL_INPUT_DIM
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_streaming_matches_offline_single_push():
    rng = np.random.default_rng(9)
    audio = rng.normal(0, 0.1, 20000).astype(np.float32)
    offline = extract_logmel23_cummn_offline(audio)
    frontend = StreamingLSEENDFrontend()
    emitted = frontend.push_audio(audio)
    tail = frontend.finalize()
    frames = (
        np.concatenate([emitted, tail], axis=0)
        if emitted.size and tail.size
        else (emitted if emitted.size else tail)
    )
    aligned = min(offline.shape[0], frames.shape[0])
    assert aligned > 0
    assert np.abs(offline[:aligned] - frames[:aligned]).max() < 1e-4


def test_streaming_chunked_matches_whole_file():
    rng = np.random.default_rng(17)
    audio = rng.normal(0, 0.1, 32173).astype(np.float32)
    whole = stream_whole_file(audio)
    resampler = Resampler16k8k()
    frontend = StreamingLSEENDFrontend()
    collected = []
    offset = 0
    while offset < audio.size:
        emitted = frontend.push_audio(resampler.push(audio[offset : offset + 512]))
        if emitted.size:
            collected.append(emitted)
        offset += 512
    tail = frontend.finalize()
    if tail.size:
        collected.append(tail)
    chunked = (
        np.concatenate(collected)
        if collected
        else np.zeros((0, LS_EEND_MODEL_INPUT_DIM), np.float32)
    )
    aligned = min(whole.shape[0], chunked.shape[0])
    assert aligned > 0
    assert whole.shape[0] == chunked.shape[0]
    assert np.abs(whole[:aligned] - chunked[:aligned]).max() < 2e-5


def test_streaming_emits_frames_in_order():
    rng = np.random.default_rng(4)
    audio = rng.normal(0, 0.1, 24000).astype(np.float32)
    frontend = StreamingLSEENDFrontend()
    offset = 0
    while offset < audio.size:
        emitted = frontend.push_audio(audio[offset : offset + 1000])
        assert emitted.shape[1] == LS_EEND_MODEL_INPUT_DIM
        offset += 1000
    assert frontend.next_model_frame == model_frame_count_offline(audio.size)


def test_frontend_frame_rate():
    assert 8000 / (LS_EEND_FRAME_HOP_8K * LS_EEND_SUBSAMPLING) == 10.0


def test_pinned_resampler_taps():
    from experiments.speaker_turn_boundary.frontend import halfband_decimation_taps

    taps = halfband_decimation_taps()
    assert taps.size == RESAMPLER_TAPS
    assert abs(float(taps.sum()) - 1.0) < 1e-12
    assert taps[RESAMPLER_CENTER_16K] == pytest.approx(0.5, abs=0.01)
