from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import medfilt

from experiments.speaker_turn_boundary.frontend import output_frame_center_16k

ReductionPolicy = Literal["new_speaker_onset", "dominant_replacement"]


@dataclass(frozen=True, slots=True)
class ReductionProfile:
    threshold: float
    persistence: int
    policy: ReductionPolicy
    median_width: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.persistence < 1:
            raise ValueError(f"persistence must be >= 1, got {self.persistence}")
        if self.median_width < 1 or self.median_width % 2 == 0:
            raise ValueError(f"median_width must be odd and >= 1, got {self.median_width}")
        if self.policy not in ("new_speaker_onset", "dominant_replacement"):
            raise ValueError(f"unknown policy {self.policy}")

    @property
    def median_shift(self) -> int:
        return self.median_width // 2

    @property
    def profile_id(self) -> str:
        return f"{self.policy}@thr{self.threshold:.2f}-p{self.persistence}-med{self.median_width}"

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "persistence": self.persistence,
            "policy": self.policy,
            "median_width": self.median_width,
        }


@dataclass(frozen=True, slots=True)
class ReducedBoundary:
    audio_epoch: int
    onset_output_frame: int
    confirmed_output_frame: int
    track_index: int | None
    confidence: float
    debug: dict[str, object]

    def boundary_source_sample(self) -> int:
        return output_frame_center_16k(self.onset_output_frame)


def batch_binary_decisions(probabilities: np.ndarray, profile: ReductionProfile) -> np.ndarray:
    binary = (probabilities > profile.threshold).astype(np.float32)
    if profile.median_width > 1:
        binary = medfilt(binary, kernel_size=(profile.median_width, 1)).astype(np.float32)
    return binary


def _mean_probability(probabilities: np.ndarray, onset: int, persistence: int, track: int) -> float:
    return float(np.mean(probabilities[onset : onset + persistence, track]))


def batch_reduce(
    probabilities: np.ndarray,
    profile: ReductionProfile,
    *,
    audio_epoch: int = 0,
) -> list[ReducedBoundary]:
    binary = batch_binary_decisions(probabilities, profile)
    frame_count, track_count = binary.shape
    boundaries: list[ReducedBoundary] = []
    if profile.policy == "new_speaker_onset":
        for track in range(track_count):
            run_start: int | None = None
            for frame in range(frame_count):
                if binary[frame, track] == 1.0:
                    if run_start is None:
                        run_start = frame
                    run_length = frame - run_start + 1
                    if run_length == profile.persistence and run_start != 0:
                        boundaries.append(
                            ReducedBoundary(
                                audio_epoch=audio_epoch,
                                onset_output_frame=run_start,
                                confirmed_output_frame=run_start + profile.persistence - 1,
                                track_index=track,
                                confidence=_mean_probability(
                                    probabilities, run_start, profile.persistence, track
                                ),
                                debug={
                                    "policy": profile.policy,
                                    "median_shift": profile.median_shift,
                                },
                            )
                        )
                else:
                    run_start = None
    else:
        previous: int | None = None
        run_start: int | None = None
        for frame in range(frame_count):
            active_tracks = [track for track in range(track_count) if binary[frame, track] == 1.0]
            if not active_tracks:
                previous = None
                run_start = None
                continue
            dominant = max(active_tracks, key=lambda track: (probabilities[frame, track], -track))
            if previous is None:
                previous = dominant
                continue
            if dominant == previous:
                if run_start is not None and frame - run_start + 1 >= profile.persistence:
                    boundaries.append(
                        ReducedBoundary(
                            audio_epoch=audio_epoch,
                            onset_output_frame=run_start,
                            confirmed_output_frame=frame,
                            track_index=dominant,
                            confidence=_mean_probability(
                                probabilities, run_start, profile.persistence, dominant
                            ),
                            debug={
                                "policy": profile.policy,
                                "median_shift": profile.median_shift,
                                "previous_dominant": previous,
                            },
                        )
                    )
                    run_start = None
                continue
            if binary[frame, previous] == 1.0:
                previous = dominant
                run_start = None
                continue
            if run_start is None:
                run_start = frame
            if frame - run_start + 1 >= profile.persistence:
                boundaries.append(
                    ReducedBoundary(
                        audio_epoch=audio_epoch,
                        onset_output_frame=run_start,
                        confirmed_output_frame=run_start + profile.persistence - 1,
                        track_index=dominant,
                        confidence=_mean_probability(
                            probabilities, run_start, profile.persistence, dominant
                        ),
                        debug={
                            "policy": profile.policy,
                            "median_shift": profile.median_shift,
                            "previous_dominant": previous,
                        },
                    )
                )
                run_start = None
            previous = dominant
    boundaries.sort(
        key=lambda item: (
            item.onset_output_frame,
            -1 if item.track_index is None else item.track_index,
        )
    )
    return boundaries


class StreamingReducer:
    def __init__(
        self,
        profile: ReductionProfile,
        *,
        track_count: int,
        audio_epoch: int,
        sample_count_at_epoch_end: int,
    ) -> None:
        self._profile = profile
        self._track_count = track_count
        self._audio_epoch = audio_epoch
        self._sample_count_at_epoch_end = sample_count_at_epoch_end
        self._median_shift = profile.median_shift
        self._raw_binary: list[np.ndarray] = []
        self._probabilities: list[np.ndarray] = []
        self._decided: list[np.ndarray] = []
        self._decided_available = -1
        self._onset_runs: dict[int, int | None] = {}
        self._dominant_run_start: int | None = None
        self._previous_dominant: int | None = None
        self._processed_frame = -1
        self._boundaries: list[ReducedBoundary] = []
        self._finalized = False

    @property
    def profile(self) -> ReductionProfile:
        return self._profile

    @property
    def processed_frame(self) -> int:
        return self._processed_frame

    def emit(self, frame: int, probabilities: np.ndarray) -> None:
        if self._finalized:
            raise RuntimeError("reducer already finalized")
        if frame != self._processed_frame + 1:
            raise ValueError(
                f"frames must arrive in order, got {frame} after {self._processed_frame}"
            )
        probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if probabilities.size != self._track_count:
            raise ValueError(
                f"expected {self._track_count} track probabilities, got {probabilities.size}"
            )
        self._probabilities.append(probabilities)
        self._raw_binary.append((probabilities > self._profile.threshold).astype(np.float32))
        self._processed_frame = frame
        self._advance_decided()

    def emit_final_tail(
        self, probabilities: np.ndarray, *, epoch_end_count: int | None = None
    ) -> None:
        if self._finalized:
            raise RuntimeError("reducer already finalized")
        if epoch_end_count is not None:
            self._sample_count_at_epoch_end = epoch_end_count
        probabilities = np.asarray(probabilities, dtype=np.float32)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)
        if probabilities.shape[1] != self._track_count:
            raise ValueError(
                f"expected {self._track_count} track probabilities, got {probabilities.shape[1]}"
            )
        for frame_probabilities in probabilities:
            self._probabilities.append(frame_probabilities)
            self._raw_binary.append(
                (frame_probabilities > self._profile.threshold).astype(np.float32)
            )
            self._processed_frame += 1
        self.finalize()

    def finalize(self, *, epoch_end_count: int | None = None) -> None:
        if self._finalized:
            return
        if epoch_end_count is not None:
            self._sample_count_at_epoch_end = epoch_end_count
        self._finalized = True
        while self._decided_available < len(self._raw_binary) - 1:
            self._advance_decided()

    @property
    def boundaries(self) -> list[ReducedBoundary]:
        return list(self._boundaries)

    def safe_boundary_frontier_sample(self) -> int:
        if self._finalized:
            return self._sample_count_at_epoch_end
        if self._decided_available < 0:
            return 0
        earliest_possible_onset = max(0, self._decided_available - self._profile.persistence + 2)
        return output_frame_center_16k(earliest_possible_onset)

    def _advance_decided(self) -> None:
        while self._decided_available < len(self._raw_binary) - 1:
            frame = self._decided_available + 1
            if self._median_shift > 0:
                missing = self._median_shift - (len(self._raw_binary) - 1 - frame)
                if missing > 0 and not self._finalized:
                    return
            self._decided_available = frame
            if self._median_shift == 0:
                decision = self._raw_binary[frame]
            else:
                window_frames = self._raw_binary[
                    max(0, frame - self._median_shift) : frame + self._median_shift + 1
                ]
                missing_left = max(0, self._median_shift - frame)
                missing_right = 2 * self._median_shift + 1 - len(window_frames) - missing_left
                if missing_left or missing_right:
                    window_frames = (
                        [np.zeros(self._track_count, dtype=np.float32)] * missing_left
                        + window_frames
                        + [np.zeros(self._track_count, dtype=np.float32)] * missing_right
                    )
                decision = np.median(np.stack(window_frames), axis=0).astype(np.float32)
            self._decided.append(decision)
            self._check_decided(frame)

    def _check_decided(self, frame: int) -> None:
        decision = self._decided[frame]
        if self._profile.policy == "new_speaker_onset":
            for track in range(self._track_count):
                if decision[track] != 1.0:
                    self._onset_runs[track] = None
                    continue
                if self._onset_runs.get(track) is None:
                    self._onset_runs[track] = frame
                onset = self._onset_runs[track]
                run_length = frame - onset + 1
                if run_length == self._profile.persistence and onset != 0:
                    self._boundaries.append(
                        ReducedBoundary(
                            audio_epoch=self._audio_epoch,
                            onset_output_frame=onset,
                            confirmed_output_frame=frame,
                            track_index=track,
                            confidence=float(
                                np.mean(
                                    [
                                        self._probabilities[item][track]
                                        for item in range(onset, onset + self._profile.persistence)
                                    ]
                                )
                            ),
                            debug={
                                "policy": self._profile.policy,
                                "median_shift": self._median_shift,
                            },
                        )
                    )
        else:
            active_tracks = [track for track in range(self._track_count) if decision[track] == 1.0]
            if not active_tracks:
                self._previous_dominant = None
                self._dominant_run_start = None
                return
            dominant = max(
                active_tracks,
                key=lambda track: (self._probabilities[frame][track], -track),
            )
            if self._previous_dominant is None:
                self._previous_dominant = dominant
                return
            if dominant == self._previous_dominant:
                if (
                    self._dominant_run_start is not None
                    and frame - self._dominant_run_start + 1 >= self._profile.persistence
                ):
                    onset = self._dominant_run_start
                    self._boundaries.append(
                        ReducedBoundary(
                            audio_epoch=self._audio_epoch,
                            onset_output_frame=onset,
                            confirmed_output_frame=frame,
                            track_index=dominant,
                            confidence=float(
                                np.mean(
                                    [
                                        self._probabilities[item][dominant]
                                        for item in range(onset, onset + self._profile.persistence)
                                    ]
                                )
                            ),
                            debug={
                                "policy": self._profile.policy,
                                "median_shift": self._median_shift,
                                "previous_dominant": self._previous_dominant,
                            },
                        )
                    )
                    self._dominant_run_start = None
                return
            if decision[self._previous_dominant] == 1.0:
                self._previous_dominant = dominant
                self._dominant_run_start = None
                return
            if self._dominant_run_start is None:
                self._dominant_run_start = frame
            if frame - self._dominant_run_start + 1 >= self._profile.persistence:
                onset = self._dominant_run_start
                self._boundaries.append(
                    ReducedBoundary(
                        audio_epoch=self._audio_epoch,
                        onset_output_frame=onset,
                        confirmed_output_frame=frame,
                        track_index=dominant,
                        confidence=float(
                            np.mean(
                                [
                                    self._probabilities[item][dominant]
                                    for item in range(onset, onset + self._profile.persistence)
                                ]
                            )
                        ),
                        debug={
                            "policy": self._profile.policy,
                            "median_shift": self._median_shift,
                            "previous_dominant": self._previous_dominant,
                        },
                    )
                )
                self._dominant_run_start = None
            self._previous_dominant = dominant
