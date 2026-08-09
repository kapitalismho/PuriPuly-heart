from __future__ import annotations

import itertools
import random

import pytest

from experiments.speaker_turn_boundary.turn_episode.phase4_design import (
    MATCH_DURATION_WEIGHT,
    MATCH_GAP_WEIGHT,
    MATCH_MAX_FEATURE_DISTANCE,
    MATCH_MAX_PAIRS,
    MATCH_MAX_TIE,
    MATCH_STRESS_WEIGHT,
    hungarian_assignment,
    ls_candidate_valid,
)


@pytest.mark.parametrize("rows", range(1, 5))
@pytest.mark.parametrize("columns", range(1, 5))
def test_exact_integer_hungarian_matches_exhaustive(rows: int, columns: int) -> None:
    for seed in range(20):
        rng = random.Random(f"{rows}:{columns}:{seed}")
        costs = [[rng.randrange(20) for _ in range(columns)] for _ in range(rows)]
        assignment = hungarian_assignment(costs)
        actual = sum(costs[row][column] for row, column in assignment)
        if rows <= columns:
            expected = min(
                sum(costs[row][choice[row]] for row in range(rows))
                for choice in itertools.permutations(range(columns), rows)
            )
        else:
            expected = min(
                sum(costs[choice[column]][column] for column in range(columns))
                for choice in itertools.permutations(range(rows), columns)
            )
        assert len(assignment) == min(rows, columns)
        assert actual == expected


def test_matching_weights_strictly_dominate_lower_order_totals() -> None:
    tie_total = MATCH_MAX_PAIRS * MATCH_MAX_TIE
    gap_total = MATCH_MAX_PAIRS * MATCH_MAX_FEATURE_DISTANCE * MATCH_GAP_WEIGHT
    duration_total = MATCH_MAX_PAIRS * MATCH_MAX_FEATURE_DISTANCE * MATCH_DURATION_WEIGHT
    assert MATCH_GAP_WEIGHT > tie_total
    assert MATCH_DURATION_WEIGHT > gap_total + tie_total
    assert MATCH_STRESS_WEIGHT > duration_total + gap_total + tie_total


@pytest.mark.parametrize("horizon_ms,support", [(250, 4000), (500, 8000), (1000, 16000)])
def test_reference_aligned_ls_acoustic_support_accepts_offset_candidate(
    horizon_ms: int, support: int
) -> None:
    candidate = {"coordinate": 32000}
    episode = {"bounds": {"warm_start": 0, "tail_end": 100000}}
    assert ls_candidate_valid(candidate, episode, horizon_ms, support)
