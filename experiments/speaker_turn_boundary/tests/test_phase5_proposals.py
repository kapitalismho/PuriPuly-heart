from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.speaker_turn_boundary.turn_episode.phase5_proposals import (
    Phase5ProposalError,
    generate_proposal_trace,
    source_prefix_routes,
)


def vector(*values: float) -> np.ndarray:
    result = np.zeros(192, dtype=np.float32)
    result[: len(values)] = values
    return result


def adjacent_profile(confirmation: int | str) -> dict[str, object]:
    return {
        "proposal_profile_id": f"adjacent:{confirmation}",
        "origin": (
            "accepted_phase4_native_profile"
            if isinstance(confirmation, str)
            else "historical_phase3_profile"
        ),
        "family": "eres2netv2",
        "checkpoint": "E-standard",
        "profile_class": "adjacent",
        "window_samples": 1000,
        "step_samples": 1000,
        "proposal_threshold": {"value": 0.5},
        "confirmation": confirmation,
        "confidence_semantics_id": "change.v1",
        "scored_state_mode": "episode_reset",
    }


def anchor_profile(profile_class: str, confirmation: int | str) -> dict[str, object]:
    historical = profile_class == "stable_anchor"
    return {
        "proposal_profile_id": f"anchor:{profile_class}:{confirmation}",
        "origin": ("historical_phase3_profile" if historical else "accepted_phase4_native_profile"),
        "family": "eres2netv2",
        "checkpoint": "E-standard",
        "profile_class": profile_class,
        "window_samples": 1000,
        "step_samples": 1000,
        "proposal_threshold": {"value": 0.5},
        "confirmation": confirmation,
        "confidence_semantics_id": "change.v1",
        "scored_state_mode": "source_prefix",
        "anchor_update": "none",
        "anchor_ema_alpha": 0.9,
        "mutual_similarity_threshold": 0.5,
    }


def episode(warm_start: int = 0, tail_end: int = 5000) -> dict[str, object]:
    return {
        "session_id": "source",
        "audio_epoch": 3,
        "bounds": {
            "warm_start": warm_start,
            "scored_start": warm_start + 1000,
            "scored_end": tail_end - 1000,
            "tail_end": tail_end,
        },
    }


def windows(values: list[np.ndarray]) -> dict[tuple[int, int], np.ndarray]:
    return {(index * 1000, (index + 1) * 1000): value for index, value in enumerate(values)}


def test_adjacent_confirmation_two_is_causal_and_latched() -> None:
    trace = generate_proposal_trace(
        windows(
            [
                vector(1, 0),
                vector(0, 1),
                vector(1, 0),
                vector(0, 1),
                vector(0, 1),
            ]
        ),
        adjacent_profile(2),
        episode(),
    )
    assert trace["proposal_count"] == 1
    event = trace["proposals"][0]
    assert event["boundary_source_sample"] == 1000
    assert event["observed_source_sample_at_emit"] == 3000
    assert event["confidence"] == pytest.approx(1.0)


def test_adjacent_native_direct_emits_every_qualifying_probe() -> None:
    trace = generate_proposal_trace(
        windows(
            [
                vector(1, 0),
                vector(0, 1),
                vector(1, 0),
                vector(0, 1),
                vector(0, 1),
            ]
        ),
        adjacent_profile("direct_each_qualifying_probe"),
        episode(),
    )
    assert [row["boundary_source_sample"] for row in trace["proposals"]] == [
        1000,
        2000,
        3000,
    ]


def test_historical_anchor_confirmation_crosses_no_vad_reset() -> None:
    trace = generate_proposal_trace(
        windows(
            [
                vector(1, 0),
                vector(0, 1),
                vector(0, 1),
                vector(0, 1),
                vector(0, 1),
            ]
        ),
        anchor_profile("stable_anchor", 2),
        episode(),
    )
    assert trace["proposal_count"] == 1
    event = trace["proposals"][0]
    assert event["boundary_source_sample"] == 1000
    assert event["observed_source_sample_at_emit"] == 3000
    assert event["state_provenance"].startswith("source_prefix:")
    assert event["debug_evidence"]["state_provenance_evidence"]["mode"] == "source_prefix"


def test_source_prefix_prewarms_without_emitting_pre_episode_action() -> None:
    trace = generate_proposal_trace(
        windows(
            [
                vector(1, 0),
                vector(0, 1),
                vector(0, 1),
                vector(0, 1),
                vector(0, 1),
            ]
        ),
        anchor_profile("stable_anchor", 2),
        episode(warm_start=2000),
    )
    assert trace["proposal_count"] == 0
    assert trace["progress"][0]["observed_source_sample"] >= 2000


def test_source_prefix_dag_routes_match_independent_replay() -> None:
    embedding_rows = windows(
        [
            vector(1, 0),
            vector(0, 1),
            vector(0, 1),
            vector(1, 0),
            vector(0, 1),
            vector(0, 1),
            vector(1, 0),
        ]
    )
    first = episode(warm_start=2000, tail_end=5000)
    first["episode_id"] = "episode:first"
    second = episode(warm_start=4000, tail_end=7000)
    second["episode_id"] = "episode:second"
    second["audio_epoch"] = 4
    profile = anchor_profile("stable_anchor", 2)
    routed = source_prefix_routes(embedding_rows, profile, [first, second])
    by_episode = {row["episode_id"]: row for row in routed["routes"]}
    for current in (first, second):
        expected = generate_proposal_trace(embedding_rows, profile, current)
        actual = by_episode[str(current["episode_id"])]
        for key in (
            "proposals",
            "progress",
            "proposal_count",
            "proposal_trace_sha256",
            "progress_trace_sha256",
            "final_state_sha256",
            "tail_evidence",
        ):
            assert actual[key] == expected[key]


def test_change_threshold_is_strict_at_exact_half() -> None:
    half = vector(0.5, math.sqrt(0.75))
    trace = generate_proposal_trace(
        windows([vector(1, 0), half, half, half, half]),
        adjacent_profile(1),
        episode(),
    )
    assert trace["proposal_count"] == 0


def test_missing_embedding_fails_closed() -> None:
    with pytest.raises(Phase5ProposalError, match="missing"):
        generate_proposal_trace(
            windows([vector(1, 0)]),
            adjacent_profile(1),
            episode(),
        )


def test_progress_frontiers_are_monotonic() -> None:
    trace = generate_proposal_trace(
        windows(
            [
                vector(1, 0),
                vector(0, 1),
                vector(1, 0),
                vector(0, 1),
                vector(0, 1),
            ]
        ),
        adjacent_profile(2),
        episode(),
    )
    observed = [row["observed_source_sample"] for row in trace["progress"]]
    safe = [row["safe_boundary_frontier_sample"] for row in trace["progress"]]
    assert observed == sorted(observed)
    assert safe == sorted(safe)
    assert all(left <= right for left, right in zip(safe, observed))
