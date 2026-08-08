from __future__ import annotations

import numpy as np

from experiments.speaker_turn_boundary.adapters.eres2netv2 import (
    EresAdjacentProfile,
    EresStableAnchorProfile,
)
from experiments.speaker_turn_boundary.events import SpeakerBoundaryEvent
from experiments.speaker_turn_boundary.ground_truth import SpeakerChangeGT
from experiments.speaker_turn_boundary.phase3_eres import (
    cached_adjacent_events,
    cached_anchor_events,
)
from experiments.speaker_turn_boundary.phase3_funnel import freeze_panel
from experiments.speaker_turn_boundary.phase3_metrics import (
    causal_ordered_match,
    causal_series,
    events_as_cuts,
    locked_product_series,
)


def _change(sample: int) -> SpeakerChangeGT:
    return SpeakerChangeGT(
        audio_epoch=0,
        change_sample=sample,
        kind="clean_handoff",
        prev_speakers=frozenset({"a"}),
        next_speakers=frozenset({"b"}),
    )


def _event(boundary: int, observed: int, source: str = "test") -> SpeakerBoundaryEvent:
    return SpeakerBoundaryEvent(
        audio_epoch=0,
        boundary_source_sample=boundary,
        observed_source_sample_at_emit=observed,
        emitted_monotonic_ns=0,
        confidence=0.8,
        source=source,
        debug={},
    )


def _vector(axis: int) -> np.ndarray:
    vector = np.zeros(192, dtype=np.float32)
    vector[axis] = 1.0
    return vector


def test_causal_match_rejects_prediction_observed_before_change() -> None:
    change = _change(16000)
    anticipatory = events_as_cuts([_event(15000, 15999)], kind="detector")
    assert causal_ordered_match([change], anticipatory, deadline_ms=2000) == []
    retrospective = events_as_cuts([_event(15200, 16800)], kind="detector")
    matches = causal_ordered_match([change], retrospective, deadline_ms=250)
    assert len(matches) == 1
    assert matches[0].observation_delay_samples == 800
    assert matches[0].localization_error_samples == -800


def test_causal_match_separates_localization_from_deadline() -> None:
    change = _change(16000)
    late_available = events_as_cuts([_event(16000, 32000)], kind="detector")
    assert causal_ordered_match([change], late_available, deadline_ms=500) == []
    assert len(causal_ordered_match([change], late_available, deadline_ms=1000)) == 1
    badly_localized = events_as_cuts([_event(25000, 26000)], kind="detector")
    assert causal_ordered_match([change], badly_localized, deadline_ms=2000) == []


def test_locked_product_preserves_b0_and_only_recovers_misses() -> None:
    changes = [_change(16000), _change(40000)]
    b0_cuts = events_as_cuts([_event(16000, 16512, "vad")], kind="vad")
    detector_cuts = events_as_cuts([_event(40000, 44000, "detector")], kind="detector")
    b0 = causal_series(changes, b0_cuts)
    product = locked_product_series(changes, b0, detector_cuts)
    assert {match.gt_index for match in product[500]} == {0, 1}
    assert len(product[500]) == len(b0[500]) + 1


def test_adjacent_confirmation_availability_is_exact() -> None:
    embeddings = {
        (0, 8000): _vector(0),
        (8000, 16000): _vector(1),
        (16000, 24000): _vector(0),
    }
    c1 = EresAdjacentProfile(
        window_seconds=0.5,
        step_seconds=0.5,
        threshold=0.5,
        confirmation=1,
    )
    events_c1, _ = cached_adjacent_events(
        utterances=[(0, 24000)],
        embeddings=embeddings,
        profile=c1,
        audio_epoch=0,
    )
    assert [
        (event.boundary_source_sample, event.observed_source_sample_at_emit) for event in events_c1
    ] == [
        (8000, 16000),
    ]
    assert all(event.confidence == 1.0 for event in events_c1)
    c2 = EresAdjacentProfile(
        window_seconds=0.5,
        step_seconds=0.5,
        threshold=0.5,
        confirmation=2,
    )
    events_c2, _ = cached_adjacent_events(
        utterances=[(0, 24000)],
        embeddings=embeddings,
        profile=c2,
        audio_epoch=0,
    )
    assert len(events_c2) == 1
    assert events_c2[0].boundary_source_sample == 8000
    assert events_c2[0].observed_source_sample_at_emit == 24000


def test_anchor_confirmation_promotes_at_real_availability() -> None:
    embeddings = {
        (0, 8000): _vector(0),
        (8000, 16000): _vector(1),
        (16000, 24000): _vector(1),
    }
    c1 = EresStableAnchorProfile(
        window_seconds=0.5,
        step_seconds=0.5,
        threshold=0.5,
        confirmation=1,
        mutual_similarity_threshold=0.5,
        anchor_update="none",
    )
    events_c1, _ = cached_anchor_events(
        utterances=[(0, 24000)],
        embeddings=embeddings,
        profile=c1,
        audio_epoch=0,
    )
    assert len(events_c1) == 1
    assert events_c1[0].boundary_source_sample == 8000
    assert events_c1[0].observed_source_sample_at_emit == 16000
    c2 = EresStableAnchorProfile(
        window_seconds=0.5,
        step_seconds=0.5,
        threshold=0.5,
        confirmation=2,
        mutual_similarity_threshold=0.5,
        anchor_update="none",
    )
    events_c2, _ = cached_anchor_events(
        utterances=[(0, 24000)],
        embeddings=embeddings,
        profile=c2,
        audio_epoch=0,
    )
    assert len(events_c2) == 1
    assert events_c2[0].boundary_source_sample == 8000
    assert events_c2[0].observed_source_sample_at_emit == 24000


def test_frozen_representative_uses_efficiency_without_a_cost_cap() -> None:
    def row(profile_id: str, recovered: int, false_cuts: int) -> dict[str, object]:
        return {
            "profile_id": profile_id,
            "family": "family",
            "checkpoint": "checkpoint",
            "profile_kind": "kind",
            "params": {},
            "extra_false_cuts": false_cuts,
            "recovered_b0_misses_at_ms": {
                "250": recovered,
                "500": recovered,
                "1000": recovered,
                "1500": recovered,
                "2000": recovered,
            },
            "timing": {},
        }

    panel = freeze_panel(
        [
            row("low-efficiency", 2, 4),
            row("best-efficiency", 4, 4),
            row("maximum-recovery", 20, 100),
        ]
    )
    assert [item["profile_id"] for item in panel] == ["best-efficiency"]
