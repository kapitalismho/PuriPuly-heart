from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_N_FFT = 400
_HOP_LENGTH = 160
_N_MELS = 80
_SAMPLE_RATE_HZ = 16000
_MEL_FLOOR = 1e-10
_NORMALIZATION_EPSILON = 1e-7


def _hertz_to_mel(frequency: np.ndarray) -> np.ndarray:
    frequency = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    log_step = 27.0 / np.log(6.4)
    mel = 3.0 * frequency / 200.0
    logarithmic = frequency >= min_log_hertz
    mel[logarithmic] = min_log_mel + np.log(frequency[logarithmic] / min_log_hertz) * log_step
    return mel


def _mel_to_hertz(mel: np.ndarray) -> np.ndarray:
    mel = np.atleast_1d(np.asarray(mel, dtype=np.float64))
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    log_step = np.log(6.4) / 27.0
    frequency = 200.0 * mel / 3.0
    logarithmic = mel >= min_log_mel
    frequency[logarithmic] = min_log_hertz * np.exp(log_step * (mel[logarithmic] - min_log_mel))
    return frequency


def _build_mel_filters() -> np.ndarray:
    frequency_bins = _N_FFT // 2 + 1
    mel_min = float(_hertz_to_mel(np.array([0.0]))[0])
    mel_max = float(_hertz_to_mel(np.array([_SAMPLE_RATE_HZ / 2.0]))[0])
    mel_points = np.linspace(mel_min, mel_max, _N_MELS + 2)
    filter_frequencies = _mel_to_hertz(mel_points)
    fft_frequencies = np.linspace(0.0, _SAMPLE_RATE_HZ / 2.0, frequency_bins)
    filter_differences = np.diff(filter_frequencies)
    slopes = filter_frequencies[None, :] - fft_frequencies[:, None]
    down_slopes = -slopes[:, :-2] / filter_differences[:-1]
    up_slopes = slopes[:, 2:] / filter_differences[1:]
    filters = np.maximum(0.0, np.minimum(down_slopes, up_slopes))
    normalization = 2.0 / (filter_frequencies[2 : _N_MELS + 2] - filter_frequencies[:_N_MELS])
    return filters * normalization[None, :]


_HANN_WINDOW = np.hanning(_N_FFT + 1)[:-1]
_MEL_FILTERS = _build_mel_filters()


def _power_spectrogram(audio: np.ndarray) -> np.ndarray:
    padded = np.pad(
        np.asarray(audio, dtype=np.float64),
        (_N_FFT // 2, _N_FFT // 2),
        mode="reflect",
    )
    windows = sliding_window_view(padded, _N_FFT)[::_HOP_LENGTH]
    spectrum = np.fft.rfft(windows * _HANN_WINDOW.astype(np.float64), axis=-1)
    return (np.abs(spectrum) ** 2).T


def compute_whisper_log_mel_features(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"audio must be one-dimensional, got {audio.shape}")
    expected_samples = _SAMPLE_RATE_HZ * 8
    if audio.size < expected_samples:
        audio = np.pad(audio, (0, expected_samples - audio.size), mode="constant")
    elif audio.size > expected_samples:
        audio = audio[:expected_samples]
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + _NORMALIZATION_EPSILON)
    mel = np.maximum(_MEL_FLOOR, _MEL_FILTERS.T @ _power_spectrogram(audio))
    log_mel = np.log10(mel)[:, :-1]
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    return ((log_mel + 4.0) / 4.0).astype(np.float32)


__all__ = ["compute_whisper_log_mel_features"]
