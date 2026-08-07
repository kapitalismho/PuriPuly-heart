from __future__ import annotations

import pytest

from experiments.speaker_turn_boundary.timeline import (
    EpochRegistry,
    SourcePosition,
    StaleEpochError,
    TimelineError,
)


def test_source_position_milliseconds_conversion():
    position = SourcePosition(audio_epoch=0, sample_index_16k=16000)
    assert position.milliseconds == 1000.0
    assert SourcePosition(audio_epoch=0, sample_index_16k=8000).milliseconds == 500.0


def test_source_position_rejects_negative_values():
    with pytest.raises(TimelineError):
        SourcePosition(audio_epoch=-1, sample_index_16k=0).validate()
    with pytest.raises(TimelineError):
        SourcePosition(audio_epoch=0, sample_index_16k=-1).validate()


def test_source_position_rejects_out_of_bounds_sample():
    with pytest.raises(TimelineError):
        SourcePosition(audio_epoch=0, sample_index_16k=100).validate(epoch_length_samples=100)


def test_source_position_accepts_last_valid_sample():
    SourcePosition(audio_epoch=0, sample_index_16k=99).validate(epoch_length_samples=100)


def test_epoch_registry_open_requires_monotonic_increasing_epochs():
    registry = EpochRegistry()
    registry.open_epoch(0)
    registry.open_epoch(1)
    with pytest.raises(TimelineError):
        registry.open_epoch(1)
    with pytest.raises(TimelineError):
        registry.open_epoch(0)
    with pytest.raises(TimelineError):
        registry.open_epoch(-1)


def test_epoch_registry_close_finalizes_length():
    registry = EpochRegistry()
    registry.open_epoch(0)
    assert registry.epoch_length(0) is None
    registry.close_epoch(0, length_samples=16000)
    assert registry.epoch_length(0) == 16000


def test_epoch_registry_close_requires_current_epoch():
    registry = EpochRegistry()
    registry.open_epoch(0)
    with pytest.raises(TimelineError):
        registry.close_epoch(1, length_samples=10)
    with pytest.raises(TimelineError):
        registry.close_epoch(0, length_samples=-1)


def test_epoch_registry_validate_sample_before_any_epoch():
    registry = EpochRegistry()
    with pytest.raises(TimelineError):
        registry.validate_sample(0, 0)


def test_epoch_registry_validate_sample_negative():
    registry = EpochRegistry()
    registry.open_epoch(0)
    with pytest.raises(TimelineError):
        registry.validate_sample(0, -1)


def test_epoch_registry_validate_sample_out_of_bounds_after_close():
    registry = EpochRegistry()
    registry.open_epoch(0)
    registry.close_epoch(0, length_samples=8000)
    with pytest.raises(TimelineError):
        registry.validate_sample(0, 8000)
    registry.validate_sample(0, 7999)


def test_epoch_registry_validate_sample_unknown_future_epoch():
    registry = EpochRegistry()
    registry.open_epoch(0)
    with pytest.raises(TimelineError):
        registry.validate_sample(1, 0)


def test_epoch_registry_marks_older_epoch_as_stale():
    registry = EpochRegistry()
    registry.open_epoch(0)
    registry.close_epoch(0, length_samples=8000)
    registry.validate_sample(0, 100)
    registry.open_epoch(1)
    registry.close_epoch(1, length_samples=8000)
    registry.validate_sample(1, 100)
    with pytest.raises(StaleEpochError):
        registry.validate_sample(0, 100)


def test_epoch_registry_validate_position_delegates_to_sample_validation():
    registry = EpochRegistry()
    registry.open_epoch(0)
    registry.validate_position(SourcePosition(audio_epoch=0, sample_index_16k=100))
    with pytest.raises(TimelineError):
        registry.validate_position(SourcePosition(audio_epoch=0, sample_index_16k=-1))
