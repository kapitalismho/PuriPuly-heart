from __future__ import annotations

from dataclasses import dataclass

from experiments.speaker_turn_boundary.config import CANONICAL_SAMPLE_RATE_HZ


class TimelineError(ValueError):
    pass


class StaleEpochError(TimelineError):
    pass


@dataclass(frozen=True, slots=True)
class SourcePosition:
    audio_epoch: int
    sample_index_16k: int

    @property
    def milliseconds(self) -> float:
        return self.sample_index_16k / (CANONICAL_SAMPLE_RATE_HZ / 1000.0)

    def validate(self, *, epoch_length_samples: int | None = None) -> None:
        if self.audio_epoch < 0:
            raise TimelineError(f"audio_epoch must be >= 0, got {self.audio_epoch}")
        if self.sample_index_16k < 0:
            raise TimelineError(f"sample_index_16k must be >= 0, got {self.sample_index_16k}")
        if epoch_length_samples is not None and self.sample_index_16k >= epoch_length_samples:
            raise TimelineError(
                f"sample_index_16k {self.sample_index_16k} is out of bounds for "
                f"epoch length {epoch_length_samples}"
            )


class EpochRegistry:
    def __init__(self) -> None:
        self._epoch_lengths: dict[int, int] = {}
        self._current_epoch: int | None = None

    @property
    def current_epoch(self) -> int | None:
        return self._current_epoch

    def epoch_length(self, audio_epoch: int) -> int | None:
        return self._epoch_lengths.get(audio_epoch)

    def open_epoch(self, audio_epoch: int) -> None:
        if audio_epoch < 0:
            raise TimelineError(f"audio_epoch must be >= 0, got {audio_epoch}")
        if self._current_epoch is not None and audio_epoch <= self._current_epoch:
            raise TimelineError(
                f"audio_epoch {audio_epoch} is not greater than current epoch "
                f"{self._current_epoch}"
            )
        self._current_epoch = audio_epoch

    def close_epoch(self, audio_epoch: int, *, length_samples: int) -> None:
        if audio_epoch != self._current_epoch:
            raise TimelineError(
                f"cannot close epoch {audio_epoch}; current epoch is {self._current_epoch}"
            )
        if length_samples < 0:
            raise TimelineError(f"length_samples must be >= 0, got {length_samples}")
        self._epoch_lengths[audio_epoch] = length_samples

    def validate_sample(self, audio_epoch: int, sample_index_16k: int) -> None:
        if audio_epoch < 0:
            raise TimelineError(f"audio_epoch must be >= 0, got {audio_epoch}")
        if self._current_epoch is None:
            raise TimelineError("no epoch has been opened")
        if audio_epoch < self._current_epoch:
            raise StaleEpochError(
                f"audio_epoch {audio_epoch} is stale; current epoch is " f"{self._current_epoch}"
            )
        if audio_epoch > self._current_epoch:
            raise TimelineError(
                f"audio_epoch {audio_epoch} is not open; current epoch is " f"{self._current_epoch}"
            )
        if sample_index_16k < 0:
            raise TimelineError(f"sample_index_16k must be >= 0, got {sample_index_16k}")
        length_samples = self._epoch_lengths.get(audio_epoch)
        if length_samples is not None and sample_index_16k >= length_samples:
            raise TimelineError(
                f"sample_index_16k {sample_index_16k} is out of bounds for epoch "
                f"{audio_epoch} length {length_samples}"
            )

    def validate_position(self, position: SourcePosition) -> None:
        self.validate_sample(position.audio_epoch, position.sample_index_16k)
