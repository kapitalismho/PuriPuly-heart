from __future__ import annotations

import pytest

from experiments.speaker_turn_boundary.events import (
    DetectorProgress,
    EventValidationError,
    SpeakerBoundaryEvent,
)


def make_event(**overrides) -> SpeakerBoundaryEvent:
    fields = {
        "audio_epoch": 0,
        "boundary_source_sample": 100,
        "observed_source_sample_at_emit": 200,
        "emitted_monotonic_ns": 1000,
        "confidence": None,
        "source": "fake_detector",
        "debug": {"k": "v"},
    }
    fields.update(overrides)
    return SpeakerBoundaryEvent(**fields)


def test_event_validation_rejects_negative_fields():
    with pytest.raises(EventValidationError):
        make_event(audio_epoch=-1)
    with pytest.raises(EventValidationError):
        make_event(boundary_source_sample=-1)
    with pytest.raises(EventValidationError):
        make_event(observed_source_sample_at_emit=-1)
    with pytest.raises(EventValidationError):
        make_event(emitted_monotonic_ns=-1)


def test_event_validation_rejects_boundary_after_observed_frontier():
    with pytest.raises(EventValidationError):
        make_event(boundary_source_sample=201)


def test_event_validation_accepts_boundary_equal_to_observed_frontier():
    event = make_event(boundary_source_sample=200)
    assert event.event_lookback_ms == 0.0


def test_event_validation_rejects_invalid_confidence():
    with pytest.raises(EventValidationError):
        make_event(confidence=-0.1)
    with pytest.raises(EventValidationError):
        make_event(confidence=1.1)


def test_event_validation_rejects_empty_source():
    with pytest.raises(EventValidationError):
        make_event(source="")


def test_event_lookback_ms_formula():
    event = make_event(boundary_source_sample=100, observed_source_sample_at_emit=260)
    assert event.event_lookback_ms == pytest.approx(10.0)


def test_event_to_dict_from_dict_round_trip():
    event = make_event(debug={"z": 1, "a": 2})
    restored = SpeakerBoundaryEvent.from_dict(event.to_dict())
    assert restored == event


def test_event_to_dict_sorts_debug_keys():
    event = make_event(debug={"z": 1, "a": 2})
    assert list(event.to_dict()["debug"].items()) == [("a", 2), ("z", 1)]


def test_progress_validation_rejects_negative_and_crossing_frontier():
    with pytest.raises(EventValidationError):
        DetectorProgress(audio_epoch=0, observed_source_sample=-1, safe_boundary_frontier_sample=0)
    with pytest.raises(EventValidationError):
        DetectorProgress(
            audio_epoch=0,
            observed_source_sample=100,
            safe_boundary_frontier_sample=101,
        )
    with pytest.raises(EventValidationError):
        DetectorProgress(
            audio_epoch=-1, observed_source_sample=100, safe_boundary_frontier_sample=0
        )


def test_progress_accepts_safe_equal_to_observed():
    progress = DetectorProgress(
        audio_epoch=0, observed_source_sample=100, safe_boundary_frontier_sample=100
    )
    assert progress.safe_boundary_frontier_sample == 100


def test_progress_to_dict_from_dict_round_trip():
    progress = DetectorProgress(
        audio_epoch=0, observed_source_sample=100, safe_boundary_frontier_sample=99
    )
    restored = DetectorProgress.from_dict(progress.to_dict())
    assert restored == progress
