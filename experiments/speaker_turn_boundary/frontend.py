from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

RESAMPLER_INPUT_RATE_HZ = 16000
RESAMPLER_OUTPUT_RATE_HZ = 8000
RESAMPLER_TAPS = 63
RESAMPLER_CENTER_16K = (RESAMPLER_TAPS - 1) // 2

LS_EEND_SAMPLE_RATE_HZ = 8000
LS_EEND_FRAME_HOP_8K = 80
LS_EEND_FFT_SIZE = 1 << (200 - 1).bit_length()
LS_EEND_LEFT_PAD_8K = LS_EEND_FFT_SIZE // 2
LS_EEND_WIN_LENGTH_8K = 200
LS_EEND_N_MELS = 23
LS_EEND_CONTEXT = 7
LS_EEND_SUBSAMPLING = 10
LS_EEND_CONV_DELAY = 9
LS_EEND_MODEL_INPUT_DIM = (2 * LS_EEND_CONTEXT + 1) * LS_EEND_N_MELS
LS_EEND_LOG_MEL_FLOOR = 1e-10


def halfband_decimation_taps() -> np.ndarray:
    taps = np.zeros(RESAMPLER_TAPS, dtype=np.float64)
    for k in range(RESAMPLER_TAPS):
        x = (k - RESAMPLER_CENTER_16K) / 2.0
        sinc = math.sin(math.pi * x) / (math.pi * x) if x != 0.0 else 1.0
        window = 0.54 - 0.46 * math.cos(2.0 * math.pi * k / (RESAMPLER_TAPS - 1))
        taps[k] = sinc * window
    taps /= taps.sum()
    return taps


def mel_filterbank_8000_fft23() -> np.ndarray:
    f_0 = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_0) / f_sp
    logstep = math.log(6.4) / 27.0

    def hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
        mels = (frequencies - f_0) / f_sp
        log_t = frequencies >= min_log_hz
        mels = mels.copy()
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels: np.ndarray) -> np.ndarray:
        freqs = f_0 + f_sp * mels
        log_t = mels >= min_log_mel
        freqs = freqs.copy()
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
        return freqs

    fft_freqs = np.linspace(0.0, 8000.0 / 2.0, LS_EEND_FFT_SIZE // 2 + 1)
    mel_min = hz_to_mel(np.array([0.0]))[0]
    mel_max = hz_to_mel(np.array([8000.0 / 2.0]))[0]
    mel_points = np.linspace(mel_min, mel_max, LS_EEND_N_MELS + 2)
    mel_f = mel_to_hz(mel_points)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)
    weights = np.zeros((LS_EEND_N_MELS, LS_EEND_FFT_SIZE // 2 + 1))
    for mel_index in range(LS_EEND_N_MELS):
        lower = -ramps[mel_index] / fdiff[mel_index]
        upper = ramps[mel_index + 2] / fdiff[mel_index + 1]
        weights[mel_index] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : LS_EEND_N_MELS + 2] - mel_f[:LS_EEND_N_MELS])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


def ls_eend_window() -> np.ndarray:
    n = np.arange(LS_EEND_WIN_LENGTH_8K, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * math.pi * n / LS_EEND_WIN_LENGTH_8K)
    padded = np.zeros(LS_EEND_FFT_SIZE, dtype=np.float32)
    pad = LS_EEND_FFT_SIZE - LS_EEND_WIN_LENGTH_8K
    padded[pad // 2 : pad // 2 + LS_EEND_WIN_LENGTH_8K] = hann.astype(np.float32)
    return padded


class Resampler16k8k:
    def __init__(self) -> None:
        self._taps = halfband_decimation_taps()
        self._history = np.zeros(RESAMPLER_TAPS - 1, dtype=np.float64)
        self._input_count = 0
        self._emitted_count = 0

    @property
    def input_count(self) -> int:
        return self._input_count

    @property
    def emitted_count(self) -> int:
        return self._emitted_count

    def push(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        old_count = self._input_count
        new_count = old_count + chunk.size
        combined = np.concatenate([self._history, chunk.astype(np.float64)])
        total_available = max(0, (new_count - RESAMPLER_CENTER_16K + 1) // 2)
        output: list[np.ndarray] = []
        for m in range(self._emitted_count, total_available):
            target = 2 * m + RESAMPLER_CENTER_16K
            start_local = target - old_count
            output.append(
                np.dot(self._taps, combined[start_local : start_local + RESAMPLER_TAPS][::-1])
            )
        self._emitted_count = total_available
        if combined.size >= RESAMPLER_TAPS - 1:
            self._history = combined[-(RESAMPLER_TAPS - 1) :]
        else:
            self._history = combined
        self._input_count = new_count
        if not output:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(output, dtype=np.float32)

    def flush(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


def source_16k_sample_of_8k_sample(sample_8k: int) -> int:
    return 2 * sample_8k + RESAMPLER_CENTER_16K


def available_8k_count(frontier_16k_count: int) -> int:
    return max(0, (frontier_16k_count - RESAMPLER_CENTER_16K + 1) // 2)


def stft_frame_center_8k(frame_index: int) -> int:
    return frame_index * LS_EEND_FRAME_HOP_8K


def stft_frame_required_8k_count(frame_index: int) -> int:
    return frame_index * LS_EEND_FRAME_HOP_8K + LS_EEND_LEFT_PAD_8K


def model_input_frame_center_8k(frame_index: int) -> int:
    return stft_frame_center_8k(frame_index * LS_EEND_SUBSAMPLING)


def model_input_frame_required_8k_count(frame_index: int) -> int:
    return stft_frame_required_8k_count(frame_index * LS_EEND_SUBSAMPLING + LS_EEND_CONTEXT)


def model_input_frame_required_16k_count(frame_index: int) -> int:
    return source_16k_sample_of_8k_sample(model_input_frame_required_8k_count(frame_index) - 1) + 1


def output_frame_center_16k(output_frame: int) -> int:
    return source_16k_sample_of_8k_sample(
        model_input_frame_center_8k(output_frame + LS_EEND_CONV_DELAY)
    )


def output_frame_available_16k_count(output_frame: int) -> int:
    return model_input_frame_required_16k_count(output_frame + LS_EEND_CONV_DELAY)


def output_frame_lookback_16k() -> int:
    return output_frame_available_16k_count(0) - output_frame_center_16k(0)


def stft_frame_count_offline(total_8k_count: int) -> int:
    if total_8k_count <= 0:
        return 0
    count = 1 + total_8k_count // LS_EEND_FRAME_HOP_8K
    if total_8k_count % LS_EEND_FRAME_HOP_8K == 0:
        count -= 1
    return max(0, count)


def model_frame_count_offline(total_8k_count: int) -> int:
    stft_count = stft_frame_count_offline(total_8k_count)
    if stft_count <= 0:
        return 0
    return (stft_count + LS_EEND_SUBSAMPLING - 1) // LS_EEND_SUBSAMPLING


def extract_logmel23_cummn_offline(audio_8k: np.ndarray) -> np.ndarray:
    audio_8k = np.asarray(audio_8k, dtype=np.float32).reshape(-1)
    total_frames = stft_frame_count_offline(audio_8k.size)
    if total_frames <= 0:
        return np.zeros((0, LS_EEND_MODEL_INPUT_DIM), dtype=np.float32)
    stft = stft_center_false(_offline_segment(audio_8k, total_frames))
    transformed = transform_logmel23_cummn_full(stft)
    spliced = splice_features(transformed, LS_EEND_CONTEXT)
    return spliced[::LS_EEND_SUBSAMPLING].astype(np.float32, copy=False)


def _offline_segment(audio_8k: np.ndarray, total_frames: int) -> np.ndarray:
    global_start = -LS_EEND_LEFT_PAD_8K
    global_stop = (total_frames - 1) * LS_EEND_FRAME_HOP_8K + LS_EEND_LEFT_PAD_8K
    return np.concatenate(
        [
            np.zeros(-global_start, dtype=np.float32),
            audio_8k,
            np.zeros(max(0, global_stop - audio_8k.size), dtype=np.float32),
        ]
    )


def stft_center_false(segment: np.ndarray) -> np.ndarray:
    segment = np.asarray(segment, dtype=np.float32).reshape(-1)
    frame_count = max(0, (segment.size - LS_EEND_FFT_SIZE) // LS_EEND_FRAME_HOP_8K + 1)
    if frame_count <= 0:
        return np.zeros((0, LS_EEND_FFT_SIZE // 2 + 1), dtype=np.complex64)
    window = ls_eend_window()
    output = np.zeros((frame_count, LS_EEND_FFT_SIZE // 2 + 1), dtype=np.complex64)
    for frame_index in range(frame_count):
        start = frame_index * LS_EEND_FRAME_HOP_8K
        output[frame_index] = np.fft.rfft(window * segment[start : start + LS_EEND_FFT_SIZE])
    return output


def transform_logmel23_cummn_full(stft: np.ndarray) -> np.ndarray:
    magnitude = np.abs(stft).astype(np.float32)
    mel = np.dot(magnitude * magnitude, mel_filterbank_8000_fft23().T)
    logmel = np.log10(np.maximum(mel, LS_EEND_LOG_MEL_FLOOR))
    cumsum = np.cumsum(logmel, axis=0)
    counts = np.arange(1, logmel.shape[0] + 1, dtype=np.float32)[:, None]
    cummean = cumsum / counts
    return (logmel - cummean).astype(np.float32, copy=False)


def splice_features(features: np.ndarray, context: int) -> np.ndarray:
    frame_count, dim = features.shape
    padded = np.zeros((frame_count + 2 * context, dim), dtype=features.dtype)
    padded[context : context + frame_count] = features
    output = np.zeros((frame_count, (2 * context + 1) * dim), dtype=features.dtype)
    for offset in range(-context, context + 1):
        source = padded[context + offset : context + offset + frame_count]
        output[:, (offset + context) * dim : (offset + context + 1) * dim] = source
    return output


class StreamingLSEENDFrontend:
    def __init__(self) -> None:
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._audio_start = 0
        self._total_8k_count = 0
        self._next_stft_frame = 0
        self._next_model_frame = 0
        self._base_feature_start = 0
        self._base_features = np.zeros((0, LS_EEND_N_MELS), dtype=np.float32)
        self._cumulative_sum = np.zeros(LS_EEND_N_MELS, dtype=np.float64)
        self._finalized = False

    @property
    def total_8k_count(self) -> int:
        return self._total_8k_count

    @property
    def next_model_frame(self) -> int:
        return self._next_model_frame

    def push_audio(self, chunk_8k: np.ndarray) -> np.ndarray:
        chunk_8k = np.asarray(chunk_8k, dtype=np.float32).reshape(-1)
        if chunk_8k.size == 0:
            return np.zeros((0, LS_EEND_MODEL_INPUT_DIM), dtype=np.float32)
        self._audio_buffer = np.concatenate([self._audio_buffer, chunk_8k])
        self._total_8k_count += chunk_8k.size
        self._append_stft_frames(_stable_stft_frame_count(self._total_8k_count))
        return self._emit_model_frames()

    def finalize(self) -> np.ndarray:
        if self._finalized:
            return np.zeros((0, LS_EEND_MODEL_INPUT_DIM), dtype=np.float32)
        self._finalized = True
        total_stft = stft_frame_count_offline(self._total_8k_count)
        self._append_stft_frames_final(total_stft)
        return self._emit_model_frames_final(total_stft)

    def _append_stft_frames(self, target_count: int) -> None:
        if target_count <= self._next_stft_frame:
            return
        segment = self._stft_segment(self._next_stft_frame, target_count, right_pad=False)
        self._append_stft_segment(segment, target_count)

    def _append_stft_frames_final(self, target_count: int) -> None:
        if target_count <= self._next_stft_frame:
            return
        segment = self._stft_segment(self._next_stft_frame, target_count, right_pad=True)
        self._append_stft_segment(segment, target_count)

    def _append_stft_segment(self, segment: np.ndarray, target_count: int) -> None:
        stft = stft_center_false(segment)
        expected = target_count - self._next_stft_frame
        if stft.shape[0] < expected:
            raise RuntimeError(
                f"streaming stft underflow: expected {expected} frames, got {stft.shape[0]}"
            )
        transformed = self._transform_batch(stft[:expected], self._next_stft_frame)
        if self._base_features.size == 0:
            self._base_features = transformed
        else:
            self._base_features = np.concatenate([self._base_features, transformed], axis=0)
        self._next_stft_frame = target_count
        self._drop_consumed_audio()

    def _stft_segment(
        self,
        frame_start: int,
        frame_stop: int,
        *,
        right_pad: bool,
    ) -> np.ndarray:
        total = self._total_8k_count
        global_start = frame_start * LS_EEND_FRAME_HOP_8K - LS_EEND_LEFT_PAD_8K
        global_stop = (
            (frame_stop - 1) * LS_EEND_FRAME_HOP_8K - LS_EEND_LEFT_PAD_8K + LS_EEND_FFT_SIZE
        )
        prefix = np.zeros(max(0, -global_start), dtype=np.float32)
        suffix = (
            np.zeros(max(0, global_stop - total), dtype=np.float32)
            if right_pad
            else np.zeros(0, dtype=np.float32)
        )
        raw_start = max(0, global_start)
        raw_stop = min(total, global_stop)
        if raw_start < self._audio_start:
            raise RuntimeError(
                f"audio buffer underflow: need sample {raw_start}, buffer starts at {self._audio_start}"
            )
        local_start = raw_start - self._audio_start
        local_stop = raw_stop - self._audio_start
        core = self._audio_buffer[local_start:local_stop]
        if prefix.size == 0 and suffix.size == 0:
            return core
        return np.concatenate([prefix, core, suffix])

    def _transform_batch(self, stft: np.ndarray, frame_start: int) -> np.ndarray:
        magnitude = np.abs(stft).astype(np.float32)
        mel = np.dot(magnitude * magnitude, mel_filterbank_8000_fft23().T)
        logmel = np.log10(np.maximum(mel, LS_EEND_LOG_MEL_FLOOR)).astype(np.float64)
        counts = np.arange(frame_start + 1, frame_start + 1 + len(logmel), dtype=np.float64)[
            :, None
        ]
        cumsum = np.cumsum(logmel, axis=0) + self._cumulative_sum[None, :]
        cummean = cumsum / counts
        self._cumulative_sum = cumsum[-1]
        return (logmel - cummean).astype(np.float32, copy=False)

    def _emit_model_frames(self) -> np.ndarray:
        outputs: list[np.ndarray] = []
        latest_frame = self._next_stft_frame - 1
        while True:
            center_index = self._next_model_frame * LS_EEND_SUBSAMPLING
            if center_index + LS_EEND_CONTEXT > latest_frame:
                break
            outputs.append(self._splice_frame(center_index, latest_frame))
            self._next_model_frame += 1
            self._drop_consumed_base_features()
        if not outputs:
            return np.zeros((0, LS_EEND_MODEL_INPUT_DIM), dtype=np.float32)
        return np.stack(outputs, axis=0).astype(np.float32, copy=False)

    def _emit_model_frames_final(self, total_stft_frames: int) -> np.ndarray:
        outputs: list[np.ndarray] = []
        total_model_frames = model_frame_count_offline(self._total_8k_count)
        while self._next_model_frame < total_model_frames:
            center_index = self._next_model_frame * LS_EEND_SUBSAMPLING
            outputs.append(self._splice_frame(center_index, total_stft_frames - 1))
            self._next_model_frame += 1
            self._drop_consumed_base_features()
        if not outputs:
            return np.zeros((0, LS_EEND_MODEL_INPUT_DIM), dtype=np.float32)
        return np.stack(outputs, axis=0).astype(np.float32, copy=False)

    def _splice_frame(self, center_index: int, max_index: int) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for frame_index in range(
            center_index - LS_EEND_CONTEXT, center_index + LS_EEND_CONTEXT + 1
        ):
            if frame_index < 0 or frame_index > max_index:
                pieces.append(np.zeros(LS_EEND_N_MELS, dtype=np.float32))
                continue
            local_index = frame_index - self._base_feature_start
            if local_index < 0 or local_index >= self._base_features.shape[0]:
                raise RuntimeError(
                    f"feature buffer underflow: need frame {frame_index}, buffer covers "
                    f"[{self._base_feature_start}, {self._base_feature_start + self._base_features.shape[0] - 1}]"
                )
            pieces.append(self._base_features[local_index])
        return np.concatenate(pieces)

    def _drop_consumed_audio(self) -> None:
        keep_from = max(0, self._next_stft_frame * LS_EEND_FRAME_HOP_8K - LS_EEND_LEFT_PAD_8K)
        drop = keep_from - self._audio_start
        if drop <= 0:
            return
        self._audio_buffer = self._audio_buffer[drop:]
        self._audio_start += drop

    def _drop_consumed_base_features(self) -> None:
        keep_from = max(0, self._next_model_frame * LS_EEND_SUBSAMPLING - LS_EEND_CONTEXT)
        drop = keep_from - self._base_feature_start
        if drop <= 0:
            return
        self._base_features = self._base_features[drop:]
        self._base_feature_start += drop


def _stable_stft_frame_count(total_8k_count: int) -> int:
    if total_8k_count <= LS_EEND_LEFT_PAD_8K:
        return 0
    return max(0, (total_8k_count - LS_EEND_LEFT_PAD_8K) // LS_EEND_FRAME_HOP_8K + 1)


def stream_whole_file(audio_16k: np.ndarray) -> np.ndarray:
    resampler = Resampler16k8k()
    resampled = resampler.push(audio_16k)
    frontend = StreamingLSEENDFrontend()
    frames = frontend.push_audio(resampled)
    tail = frontend.finalize()
    if frames.size == 0 and tail.size == 0:
        return np.zeros((0, LS_EEND_MODEL_INPUT_DIM), dtype=np.float32)
    return np.concatenate([frames, tail], axis=0)


@dataclass(frozen=True, slots=True)
class FrontendProfile:
    resampler_taps: int
    resampler_center_16k: int
    sample_rate_hz: int
    win_length: int
    fft_size: int
    hop_length: int
    n_mels: int
    context_recp: int
    subsampling: int
    conv_delay: int
    feat_type: str
    frame_hz: float

    def to_dict(self) -> dict[str, object]:
        return {
            "resampler_taps": self.resampler_taps,
            "resampler_center_16k": self.resampler_center_16k,
            "sample_rate_hz": self.sample_rate_hz,
            "win_length": self.win_length,
            "fft_size": self.fft_size,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "context_recp": self.context_recp,
            "subsampling": self.subsampling,
            "conv_delay": self.conv_delay,
            "feat_type": self.feat_type,
            "frame_hz": 8000.0 / (LS_EEND_FRAME_HOP_8K * LS_EEND_SUBSAMPLING),
        }


def frontend_profile() -> FrontendProfile:
    return FrontendProfile(
        resampler_taps=RESAMPLER_TAPS,
        resampler_center_16k=RESAMPLER_CENTER_16K,
        sample_rate_hz=LS_EEND_SAMPLE_RATE_HZ,
        win_length=LS_EEND_WIN_LENGTH_8K,
        fft_size=LS_EEND_FFT_SIZE,
        hop_length=LS_EEND_FRAME_HOP_8K,
        n_mels=LS_EEND_N_MELS,
        context_recp=LS_EEND_CONTEXT,
        subsampling=LS_EEND_SUBSAMPLING,
        conv_delay=LS_EEND_CONV_DELAY,
        feat_type="logmel23_cummn",
        frame_hz=8000.0 / (LS_EEND_FRAME_HOP_8K * LS_EEND_SUBSAMPLING),
    )
