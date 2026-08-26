from __future__ import annotations

from experiments.psem_ontology_simplification_gate.rerun_s2 import (
    _aggregate_coverage,
    _canonical_match,
    _causal_records_with_coverage,
)
from experiments.psem_relative_occupancy_gate.contracts import EvaluationCell
from experiments.psem_relative_occupancy_gate.model_decode import PosteriorCell


def _cell(index: int, *, reset: bool = False) -> PosteriorCell:
    start = index * 1600
    end = start + 1600
    return PosteriorCell(
        cell=EvaluationCell(
            index=index,
            start_sample=start,
            end_sample=end,
            center_sample=start + 800,
            active_speakers=("A",),
            masked=False,
        ),
        probabilities=(0.9, 0.1),
        slot_alive=(True, True),
        evidence_frontier_sample=end,
        state_reset=reset,
        trace_valid=True,
    )


def test_s2_coverage_reports_unmapped_and_sticky_continuity_loss() -> None:
    row = {
        "source_id": "source",
        "intervals": [
            {
                "start_sample": 0,
                "end_sample": 6400,
                "active_speakers": ["A"],
                "masked": False,
            }
        ],
    }
    gate2_row = {
        "annotated_episodes": [
            {
                "episode_id": "mapped",
                "expected_anchor_speaker": "A",
                "anchor_slot_index": 0,
                "anchor_emit_sample": 0,
                "end_emit_sample": 4800,
            },
            {
                "episode_id": "unmapped",
                "expected_anchor_speaker": None,
                "anchor_slot_index": None,
                "anchor_emit_sample": 4800,
                "end_emit_sample": 6400,
            },
        ]
    }
    records, coverage = _causal_records_with_coverage(
        row,
        (_cell(0), _cell(1, reset=True), _cell(2), _cell(3)),
        gate2_row,
    )
    assert [value.start_sample for value in records] == [0]
    assert coverage == {
        "source_id": "source",
        "episode_count": 2,
        "mapped_episode_count": 1,
        "unmapped_episode_count": 1,
        "mapped_episode_fraction": 0.5,
        "episode_support_seconds": 0.4,
        "valid_diagnostic_support_seconds": 0.1,
        "unmapped_episode_support_seconds": 0.1,
        "unmapped_unmasked_active_speech_seconds": 0.1,
        "continuity_invalid_episode_count": 1,
        "continuity_invalid_support_seconds": 0.2,
        "continuity_invalid_unmasked_active_speech_seconds": 0.2,
    }


def test_s2_coverage_aggregation_preserves_per_source_evidence() -> None:
    rows = [
        {
            "source_id": "a",
            "episode_count": 2,
            "mapped_episode_count": 1,
            "unmapped_episode_count": 1,
            "mapped_episode_fraction": 0.5,
            "episode_support_seconds": 0.4,
            "valid_diagnostic_support_seconds": 0.1,
            "unmapped_episode_support_seconds": 0.1,
            "unmapped_unmasked_active_speech_seconds": 0.1,
            "continuity_invalid_episode_count": 1,
            "continuity_invalid_support_seconds": 0.2,
            "continuity_invalid_unmasked_active_speech_seconds": 0.2,
        },
        {
            "source_id": "b",
            "episode_count": 1,
            "mapped_episode_count": 1,
            "unmapped_episode_count": 0,
            "mapped_episode_fraction": 1.0,
            "episode_support_seconds": 0.1,
            "valid_diagnostic_support_seconds": 0.1,
            "unmapped_episode_support_seconds": 0.0,
            "unmapped_unmasked_active_speech_seconds": 0.0,
            "continuity_invalid_episode_count": 0,
            "continuity_invalid_support_seconds": 0.0,
            "continuity_invalid_unmasked_active_speech_seconds": 0.0,
        },
    ]
    aggregate = _aggregate_coverage(rows)
    assert aggregate["source_count"] == 2
    assert aggregate["episode_count"] == 3
    assert aggregate["mapped_episode_count"] == 2
    assert aggregate["mapped_episode_fraction"] == 2 / 3
    assert aggregate["continuity_invalid_episode_count"] == 1
    assert aggregate["continuity_invalid_support_seconds"] == 0.2
    assert aggregate["per_source"] == rows


def test_frontier_match_normalizes_json_object_keys() -> None:
    assert _canonical_match({"distribution": {1: 2}}, {"distribution": {"1": 2}})
