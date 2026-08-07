from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from experiments.speaker_turn_boundary.config import CANONICAL_SAMPLE_RATE_HZ

ERES_SAMPLE_RATE_HZ = 16000
ERES_FBANK_N_MELS = 80
ERES_FRAME_LENGTH_MS = 25.0
ERES_FRAME_SHIFT_MS = 10.0
ERES_WINDOW_SIZE = int(ERES_SAMPLE_RATE_HZ * ERES_FRAME_LENGTH_MS / 1000.0)
ERES_WINDOW_SHIFT = int(ERES_SAMPLE_RATE_HZ * ERES_FRAME_SHIFT_MS / 1000.0)
ERES_PADDED_WINDOW_SIZE = 512
ERES_PREEMPHASIS = 0.97
ERES_LOW_FREQ = 20.0
ERES_EPSILON = np.float32(np.finfo(np.float32).eps)


def _mel_scale_htk(freq: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + freq / 700.0)


def _inverse_mel_scale_htk(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.exp(mel / 1127.0) - 1.0)


def _povey_window() -> np.ndarray:
    n = np.arange(ERES_WINDOW_SIZE)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (ERES_WINDOW_SIZE - 1))
    return (hann**0.85).astype(np.float32)


def _mel_banks_htk() -> np.ndarray:
    num_fft_bins = ERES_PADDED_WINDOW_SIZE // 2
    nyquist = ERES_SAMPLE_RATE_HZ / 2.0
    fft_bin_width = ERES_SAMPLE_RATE_HZ / ERES_PADDED_WINDOW_SIZE
    mel_low = _mel_scale_htk(np.array([ERES_LOW_FREQ]))[0]
    mel_high = _mel_scale_htk(np.array([nyquist]))[0]
    mel_delta = (mel_high - mel_low) / (ERES_FBANK_N_MELS + 1)
    bin_index = np.arange(ERES_FBANK_N_MELS, dtype=np.float64)
    left_mel = mel_low + bin_index * mel_delta
    center_mel = mel_low + (bin_index + 1.0) * mel_delta
    right_mel = mel_low + (bin_index + 2.0) * mel_delta
    mel = _mel_scale_htk(fft_bin_width * np.arange(num_fft_bins))[None, :]
    up_slope = (mel - left_mel[:, None]) / (center_mel - left_mel)[:, None]
    down_slope = (right_mel[:, None] - mel) / (right_mel - center_mel)[:, None]
    bins = np.maximum(0.0, np.minimum(up_slope, down_slope))
    padded = np.zeros((ERES_FBANK_N_MELS, num_fft_bins + 1), dtype=np.float64)
    padded[:, :num_fft_bins] = bins
    return padded.astype(np.float32)


def kaldi_fbank_numpy(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size < ERES_WINDOW_SIZE:
        return np.zeros((0, ERES_FBANK_N_MELS), dtype=np.float32)
    frame_count = 1 + (waveform.size - ERES_WINDOW_SIZE) // ERES_WINDOW_SHIFT
    frames = np.zeros((frame_count, ERES_PADDED_WINDOW_SIZE), dtype=np.float32)
    for index in range(frame_count):
        start = index * ERES_WINDOW_SHIFT
        window = waveform[start : start + ERES_WINDOW_SIZE]
        window = window - np.mean(window)
        preemphasized = np.empty_like(window)
        preemphasized[0] = window[0]
        preemphasized[1:] = window[1:] - ERES_PREEMPHASIS * window[:-1]
        frames[index, :ERES_WINDOW_SIZE] = preemphasized * _povey_window()
    spectrum = np.abs(np.fft.rfft(frames, axis=1))
    power = spectrum * spectrum
    mel = np.dot(power, _mel_banks_htk().T)
    floor = np.float32(ERES_EPSILON)
    logmel = np.log(np.maximum(mel, floor))
    return logmel.astype(np.float32, copy=False)


class EresEmbeddingRuntime:
    def __init__(
        self,
        onnx_path: str,
        *,
        intra_op_threads: int = 1,
        inter_op_threads: int = 1,
    ) -> None:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = inter_op_threads
        self._session = ort.InferenceSession(
            onnx_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._output_names = [output.name for output in self._session.get_outputs()]

    def embed(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        fbank = kaldi_fbank_numpy(samples)
        if fbank.shape[0] == 0:
            return np.zeros((192,), dtype=np.float32)
        fbank = fbank - fbank.mean(axis=0, keepdims=True)
        output = self._session.run(
            self._output_names, {"fbank": fbank[None, :, :].astype(np.float32)}
        )[0]
        return np.asarray(output[0], dtype=np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    norm_product = np.linalg.norm(left) * np.linalg.norm(right)
    if norm_product == 0.0:
        return 0.0
    return float(np.dot(left, right) / norm_product)


def clamp_confidence(score: float) -> float:
    return min(1.0, max(0.0, 1.0 - float(score)))


@dataclass(frozen=True, slots=True)
class EresAdjacentProfile:
    window_seconds: float
    step_seconds: float
    threshold: float
    confirmation: int

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {self.window_seconds}")
        if self.step_seconds <= 0:
            raise ValueError(f"step_seconds must be > 0, got {self.step_seconds}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.confirmation not in (1, 2):
            raise ValueError(f"confirmation must be 1 or 2, got {self.confirmation}")

    @property
    def profile_id(self) -> str:
        return (
            f"adjacent-W{self.window_seconds:g}-s{self.step_seconds:g}"
            f"-thr{self.threshold:.2f}-c{self.confirmation}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "window_seconds": self.window_seconds,
            "step_seconds": self.step_seconds,
            "threshold": self.threshold,
            "confirmation": self.confirmation,
        }


@dataclass(frozen=True, slots=True)
class EresStableAnchorProfile:
    window_seconds: float
    step_seconds: float
    threshold: float
    confirmation: int
    mutual_similarity_threshold: float
    anchor_update: str
    anchor_ema_alpha: float = 0.9

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {self.window_seconds}")
        if self.step_seconds <= 0:
            raise ValueError(f"step_seconds must be > 0, got {self.step_seconds}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.confirmation not in (1, 2):
            raise ValueError(f"confirmation must be 1 or 2, got {self.confirmation}")
        if self.anchor_update not in ("none", "ema"):
            raise ValueError(f"anchor_update must be 'none' or 'ema', got {self.anchor_update}")

    @property
    def profile_id(self) -> str:
        return (
            f"anchor-W{self.window_seconds:g}-s{self.step_seconds:g}"
            f"-thr{self.threshold:.2f}-c{self.confirmation}"
            f"-{self.anchor_update}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "window_seconds": self.window_seconds,
            "step_seconds": self.step_seconds,
            "threshold": self.threshold,
            "confirmation": self.confirmation,
            "mutual_similarity_threshold": self.mutual_similarity_threshold,
            "anchor_update": self.anchor_update,
            "anchor_ema_alpha": self.anchor_ema_alpha,
        }


def _sample_count(seconds: float) -> int:
    return int(round(seconds * ERES_SAMPLE_RATE_HZ))


@dataclass(frozen=True, slots=True)
class EresBoundary:
    audio_epoch: int
    boundary_sample: int
    observed_sample: int
    confidence: float
    debug: dict[str, object]


class AdjacentWindowDetector:
    def __init__(
        self,
        runtime: EresEmbeddingRuntime,
        profile: EresAdjacentProfile,
        *,
        audio_epoch: int = 0,
    ) -> None:
        self._runtime = runtime
        self._profile = profile
        self._audio_epoch = audio_epoch
        self._window = _sample_count(profile.window_seconds)
        self._step = _sample_count(profile.step_seconds)

    def run_utterance(
        self,
        samples_16k: np.ndarray,
        utterance: tuple[int, int],
    ) -> list[EresBoundary]:
        start_sample, end_sample = utterance
        boundaries: list[EresBoundary] = []
        candidate: float | None = None
        candidate_position: int | None = None
        position = start_sample + self._window
        while position + self._window <= end_sample:
            left = samples_16k[position - self._window : position]
            right = samples_16k[position : position + self._window]
            score = cosine_similarity(self._runtime.embed(left), self._runtime.embed(right))
            is_candidate = score < self._profile.threshold
            if is_candidate and candidate is None:
                candidate = score
                candidate_position = position
            elif is_candidate and candidate is not None:
                if self._profile.confirmation == 2:
                    boundaries.append(
                        EresBoundary(
                            audio_epoch=self._audio_epoch,
                            boundary_sample=candidate_position,
                            observed_sample=candidate_position + self._window,
                            confidence=clamp_confidence(candidate),
                            debug={
                                "profile": self._profile.to_dict(),
                                "score_first": candidate,
                                "score_second": score,
                                "skipped_positions": 0,
                            },
                        )
                    )
                    candidate = None
                    candidate_position = None
            else:
                candidate = None
                candidate_position = None
            position += self._step
        if candidate is not None and self._profile.confirmation == 1:
            boundaries.append(
                EresBoundary(
                    audio_epoch=self._audio_epoch,
                    boundary_sample=candidate_position,
                    observed_sample=candidate_position + self._window,
                    confidence=clamp_confidence(candidate),
                    debug={
                        "profile": self._profile.to_dict(),
                        "score_first": candidate,
                        "skipped_positions": 0,
                    },
                )
            )
        return boundaries


class StableAnchorDetector:
    def __init__(
        self,
        runtime: EresEmbeddingRuntime,
        profile: EresStableAnchorProfile,
        *,
        audio_epoch: int = 0,
    ) -> None:
        self._runtime = runtime
        self._profile = profile
        self._audio_epoch = audio_epoch
        self._window = _sample_count(profile.window_seconds)
        self._step = _sample_count(profile.step_seconds)

    def run_utterance(
        self,
        samples_16k: np.ndarray,
        utterance: tuple[int, int],
    ) -> list[EresBoundary]:
        start_sample, end_sample = utterance
        boundaries: list[EresBoundary] = []
        if start_sample + self._window > end_sample:
            return boundaries
        anchor = self._runtime.embed(samples_16k[start_sample : start_sample + self._window])
        anchor = anchor / (np.linalg.norm(anchor) + 1e-12)
        candidate: float | None = None
        candidate_position: int | None = None
        candidate_embedding: np.ndarray | None = None
        position = start_sample + self._window
        while position + self._window <= end_sample:
            probe = self._runtime.embed(samples_16k[position : position + self._window])
            score = cosine_similarity(anchor, probe)
            is_candidate = score < self._profile.threshold
            if is_candidate and candidate is None:
                candidate = score
                candidate_position = position
                candidate_embedding = probe
            elif is_candidate and candidate is not None:
                confirmed = False
                if self._profile.confirmation == 1:
                    confirmed = True
                else:
                    mutual = cosine_similarity(candidate_embedding, probe)
                    confirmed = mutual >= self._profile.mutual_similarity_threshold
                if confirmed:
                    boundaries.append(
                        EresBoundary(
                            audio_epoch=self._audio_epoch,
                            boundary_sample=candidate_position,
                            observed_sample=candidate_position + self._window,
                            confidence=clamp_confidence(candidate),
                            debug={
                                "profile": self._profile.to_dict(),
                                "score_first": candidate,
                                "score_second": score,
                                "mutual_similarity": (
                                    cosine_similarity(candidate_embedding, probe)
                                    if self._profile.confirmation == 2
                                    else None
                                ),
                            },
                        )
                    )
                    anchor = probe / (np.linalg.norm(probe) + 1e-12)
                    candidate = None
                    candidate_position = None
                    candidate_embedding = None
            else:
                if (
                    self._profile.anchor_update == "ema"
                    and candidate is None
                    and np.linalg.norm(probe) > 0
                ):
                    alpha = self._profile.anchor_ema_alpha
                    anchor = (1.0 - alpha) * anchor + alpha * probe / (
                        np.linalg.norm(probe) + 1e-12
                    )
                    anchor = anchor / (np.linalg.norm(anchor) + 1e-12)
            position += self._step
        if candidate is not None and self._profile.confirmation == 1:
            boundaries.append(
                EresBoundary(
                    audio_epoch=self._audio_epoch,
                    boundary_sample=candidate_position,
                    observed_sample=candidate_position + self._window,
                    confidence=clamp_confidence(candidate),
                    debug={
                        "profile": self._profile.to_dict(),
                        "score_first": candidate,
                    },
                )
            )
        return boundaries


def eres_profile_matrix() -> list[tuple[float, list[float]]]:
    return [
        (0.50, [0.10, 0.25]),
        (0.75, [0.10, 0.25]),
        (1.00, [0.10, 0.25, 0.50]),
        (1.50, [0.25, 0.50]),
        (2.00, [0.50]),
    ]


def threshold_range() -> list[float]:
    return [round(0.30 + 0.05 * index, 2) for index in range(9)]


def eres_adjacent_profiles() -> list[EresAdjacentProfile]:
    profiles: list[EresAdjacentProfile] = []
    for window, steps in eres_profile_matrix():
        for step in steps:
            for threshold in threshold_range():
                for confirmation in (1, 2):
                    profiles.append(
                        EresAdjacentProfile(
                            window_seconds=window,
                            step_seconds=step,
                            threshold=threshold,
                            confirmation=confirmation,
                        )
                    )
    return profiles


def eres_anchor_profiles(
    *,
    windows: tuple[float, ...] = (0.50, 0.75, 1.00, 1.50),
) -> list[EresStableAnchorProfile]:
    profiles: list[EresStableAnchorProfile] = []
    for window in windows:
        for step in (0.10, 0.25):
            for threshold in threshold_range():
                for confirmation in (1, 2):
                    for anchor_update in ("none", "ema"):
                        profiles.append(
                            EresStableAnchorProfile(
                                window_seconds=window,
                                step_seconds=step,
                                threshold=threshold,
                                confirmation=confirmation,
                                mutual_similarity_threshold=0.5,
                                anchor_update=anchor_update,
                            )
                        )
    return profiles


def eres_embedding_profile_dict(embedding_size: int) -> dict[str, object]:
    return {
        "sample_rate_hz": ERES_SAMPLE_RATE_HZ,
        "fbank_n_mels": ERES_FBANK_N_MELS,
        "frame_length_ms": ERES_FRAME_LENGTH_MS,
        "frame_shift_ms": ERES_FRAME_SHIFT_MS,
        "window_type": "povey",
        "preemphasis_coefficient": ERES_PREEMPHASIS,
        "mel_scale": "htk",
        "mean_normalization": "time",
        "embedding_size": embedding_size,
    }


def eres_b0_profile_dict() -> dict[str, object]:
    return {
        "frontend": "kaldi-fbank-80-htk-povey",
        "sample_rate_hz": CANONICAL_SAMPLE_RATE_HZ,
    }
