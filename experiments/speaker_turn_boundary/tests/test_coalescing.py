from __future__ import annotations

from experiments.speaker_turn_boundary.coalescing import (
    CoalesceConfig,
    coalesce_vad_and_detector,
)
from experiments.speaker_turn_boundary.events import SpeakerBoundaryEvent


def vad_boundary(epoch: int, sample: int) -> SpeakerBoundaryEvent:
    return SpeakerBoundaryEvent(
        audio_epoch=epoch,
        boundary_source_sample=sample,
        observed_source_sample_at_emit=sample + 512,
        emitted_monotonic_ns=0,
        confidence=None,
        source="vad_b0",
        debug={},
    )


def detector_event(epoch: int, sample: int) -> SpeakerBoundaryEvent:
    return SpeakerBoundaryEvent(
        audio_epoch=epoch,
        boundary_source_sample=sample,
        observed_source_sample_at_emit=sample + 512,
        emitted_monotonic_ns=1,
        confidence=0.9,
        source="fake_detector",
        debug={},
    )


def test_detector_event_inside_window_is_coalesced():
    outcome = coalesce_vad_and_detector(
        [vad_boundary(0, 1000)],
        [detector_event(0, 1300)],
    )
    assert outcome.report.vad_cut_count == 1
    assert outcome.report.coalesced_count == 1
    assert outcome.report.duplicate_count == 0
    assert outcome.report.detector_cut_count == 0
    assert outcome.report.total_logical_cuts == 1
    assert outcome.detections[0]["disposition"] == "coalesced"
    assert outcome.detections[0]["matched_vad_sample"] == 1000
    assert len(outcome.cuts) == 1
    assert outcome.cuts[0].kind == "vad"


def test_detector_event_outside_window_creates_cut():
    window = CoalesceConfig(window_samples=8000)
    outcome = coalesce_vad_and_detector(
        [vad_boundary(0, 1000)],
        [detector_event(0, 1000 + 8000 + 1)],
        config=window,
    )
    assert outcome.report.coalesced_count == 0
    assert outcome.report.detector_cut_count == 1
    assert outcome.report.total_logical_cuts == 2
    assert outcome.detections[0]["disposition"] == "cut"
    assert outcome.detections[0]["matched_vad_sample"] is None
    assert [cut.kind for cut in outcome.cuts] == ["vad", "detector"]


def test_detector_event_at_window_boundary_is_coalesced():
    window = CoalesceConfig(window_samples=8000)
    outcome = coalesce_vad_and_detector(
        [vad_boundary(0, 1000)],
        [detector_event(0, 1000 + 8000)],
        config=window,
    )
    assert outcome.report.coalesced_count == 1
    assert outcome.report.detector_cut_count == 0


def test_detector_event_exactly_at_vad_boundary_is_coalesced():
    outcome = coalesce_vad_and_detector(
        [vad_boundary(0, 1000)],
        [detector_event(0, 1000)],
    )
    assert outcome.report.coalesced_count == 1


def test_second_event_on_same_vad_boundary_is_duplicate():
    outcome = coalesce_vad_and_detector(
        [vad_boundary(0, 1000)],
        [detector_event(0, 1100), detector_event(0, 1150)],
    )
    assert outcome.report.coalesced_count == 1
    assert outcome.report.duplicate_count == 1
    assert outcome.report.detector_cut_count == 0
    assert [d["disposition"] for d in outcome.detections] == [
        "coalesced",
        "duplicate",
    ]


def test_duplicate_does_not_fall_through_to_second_vad_boundary():
    outcome = coalesce_vad_and_detector(
        [vad_boundary(0, 1000), vad_boundary(0, 5000)],
        [detector_event(0, 1100), detector_event(0, 1300)],
    )
    assert outcome.report.coalesced_count == 1
    assert outcome.report.duplicate_count == 1
    assert outcome.report.detector_cut_count == 0


def test_stale_epoch_events_are_dropped_and_counted():
    outcome = coalesce_vad_and_detector(
        [vad_boundary(1, 1000)],
        [detector_event(0, 900), detector_event(1, 1100)],
    )
    assert outcome.report.stale_detector_events == 1
    assert outcome.report.coalesced_count == 1
    assert outcome.report.detector_cut_count == 0
    assert outcome.detections[0]["disposition"] == "stale"
    assert outcome.detections[0]["matched_vad_sample"] is None


def test_detector_epoch_without_vad_boundaries_creates_cut():
    outcome = coalesce_vad_and_detector(
        [vad_boundary(0, 1000)],
        [detector_event(2, 500)],
    )
    assert outcome.report.detector_cut_count == 1
    assert outcome.report.total_logical_cuts == 2


def test_no_vad_boundaries_all_detector_events_become_cuts():
    outcome = coalesce_vad_and_detector(
        [],
        [detector_event(0, 100), detector_event(0, 200)],
    )
    assert outcome.report.vad_cut_count == 0
    assert outcome.report.detector_cut_count == 2
    assert outcome.report.stale_detector_events == 0
    assert outcome.report.total_logical_cuts == 2


def test_matching_prefers_nearest_vad_boundary():
    outcome = coalesce_vad_and_detector(
        [vad_boundary(0, 0), vad_boundary(0, 8000)],
        [detector_event(0, 100)],
    )
    assert outcome.detections[0]["matched_vad_sample"] == 0
    assert outcome.report.coalesced_count == 1
    outcome = coalesce_vad_and_detector(
        [vad_boundary(0, 0), vad_boundary(0, 8000)],
        [detector_event(0, 7900)],
    )
    assert outcome.detections[0]["matched_vad_sample"] == 8000


def test_matching_tie_resolves_to_earlier_vad_boundary():
    outcome = coalesce_vad_and_detector(
        [vad_boundary(0, 0), vad_boundary(0, 1000)],
        [detector_event(0, 500)],
    )
    assert outcome.detections[0]["matched_vad_sample"] == 0


def test_coalescing_is_deterministic_regardless_of_arrival_order():
    vad = [vad_boundary(0, 1000), vad_boundary(0, 5000), vad_boundary(1, 1000)]
    events = [
        detector_event(0, 1100),
        detector_event(0, 5100),
        detector_event(0, 9000),
        detector_event(1, 900),
        detector_event(1, 1100),
        detector_event(0, 1200),
    ]
    first = coalesce_vad_and_detector(vad, list(events))
    shuffled = coalesce_vad_and_detector(vad, list(reversed(events)))
    assert first.report == shuffled.report
    assert first.cuts == shuffled.cuts

    def projection(outcome) -> list:
        return sorted(
            (
                item["audio_epoch"],
                item["boundary_source_sample"],
                item["disposition"],
                item["matched_vad_sample"],
            )
            for item in outcome.detections
        )

    assert projection(first) == projection(shuffled)


def test_report_and_cuts_serialize_deterministically():
    outcome = coalesce_vad_and_detector(
        [vad_boundary(0, 1000)],
        [detector_event(0, 1100), detector_event(0, 9001)],
    )
    assert outcome.report.to_dict() == {
        "vad_cut_count": 1,
        "detector_events_total": 2,
        "stale_detector_events": 0,
        "coalesced_count": 1,
        "duplicate_count": 0,
        "detector_cut_count": 1,
        "total_logical_cuts": 2,
    }
    assert outcome.to_dict()["cuts"] == [
        {"audio_epoch": 0, "sample": 1000, "kind": "vad", "ref_event_index": 0},
        {
            "audio_epoch": 0,
            "sample": 9001,
            "kind": "detector",
            "ref_event_index": 1,
        },
    ]
