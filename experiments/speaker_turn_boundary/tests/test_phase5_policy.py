from __future__ import annotations

import pytest

from experiments.speaker_turn_boundary.turn_episode.phase5_policy import (
    Phase5PolicyError,
    actionize_clusters,
    cluster_proposals,
    cluster_proposals_reference,
    frequency_matched_control,
    full_fusion_replay,
    fuse_actions,
    policy_progress,
)


def proposal(
    proposal_id: str,
    boundary: int,
    observed: int,
    confidence: float,
    *,
    kind: str = "speaker_change_unknown",
    semantics: str = "change.v1",
) -> dict[str, object]:
    return {
        "proposal_id": proposal_id,
        "family": "eres2netv2",
        "checkpoint": "E-standard",
        "profile_id": "profile",
        "audio_epoch": 7,
        "source_session_id": "source",
        "proposal_kind": kind,
        "boundary_source_sample": boundary,
        "observed_source_sample_at_emit": observed,
        "confidence": confidence,
        "confidence_semantics_id": semantics,
    }


def test_cluster_is_causal_and_refractory_is_strict() -> None:
    result = cluster_proposals(
        [
            proposal("p0", 16000, 20000, 0.6),
            proposal("p1", 16200, 21000, 0.8),
            proposal("p2", 21000, 23000, 0.7),
            proposal("p3", 27000, 27000, 0.9),
        ],
        cluster_debounce_ms=100,
        cluster_boundary_radius_ms=250,
        refractory_ms=250,
        representative="max_confidence",
        episode_observed_end=60000,
    )
    assert result["clusters"][0]["representative_proposal_id"] == "p1"
    assert result["clusters"][0]["observed_source_sample_at_emit"] == 21600
    assert [row["proposal_id"] for row in result["refractory_proposals"]] == ["p2"]
    assert result["clusters"][1]["representative_proposal_id"] == "p3"


def test_zero_refractory_preserves_queued_out_of_radius_proposal() -> None:
    result = cluster_proposals(
        [
            proposal("p0", 16000, 20000, 0.6),
            proposal("p1", 20001, 20500, 0.7),
        ],
        cluster_debounce_ms=100,
        cluster_boundary_radius_ms=250,
        refractory_ms=0,
        representative="first",
        episode_observed_end=40000,
    )
    assert [row["representative_proposal_id"] for row in result["clusters"]] == [
        "p0",
        "p1",
    ]
    assert result["clusters"][1]["cluster_open_frontier"] == 21600


def test_max_confidence_falls_back_for_incompatible_semantics() -> None:
    result = cluster_proposals(
        [
            proposal("p0", 16000, 20000, 0.2, semantics="a"),
            proposal("p1", 16100, 20100, 0.9, semantics="b"),
        ],
        cluster_debounce_ms=100,
        cluster_boundary_radius_ms=250,
        refractory_ms=0,
        representative="max_confidence",
        episode_observed_end=40000,
    )
    cluster = result["clusters"][0]
    assert cluster["representative_proposal_id"] == "p0"
    assert cluster["representative_reason"] == "first_incompatible_confidence_semantics"


def test_mixed_kind_representative_is_semantically_compatible() -> None:
    result = cluster_proposals(
        [
            proposal("hard", 16000, 20000, 0.99, kind="dominant_replacement"),
            proposal("soft", 16200, 20100, 0.20, kind="overlap_onset"),
        ],
        cluster_debounce_ms=100,
        cluster_boundary_radius_ms=250,
        refractory_ms=0,
        representative="max_confidence",
        episode_observed_end=40000,
    )
    cluster = result["clusters"][0]
    assert cluster["proposal_kind"] == "overlap_onset"
    assert cluster["representative_proposal_id"] == "soft"
    assert actionize_clusters([cluster])[0]["requested_action"] == "soft_marker"


def test_tail_close_uses_observed_end() -> None:
    result = cluster_proposals(
        [proposal("p0", 16000, 20000, 0.6)],
        cluster_debounce_ms=250,
        cluster_boundary_radius_ms=250,
        refractory_ms=0,
        representative="first",
        episode_observed_end=22000,
    )
    cluster = result["clusters"][0]
    assert cluster["tail_closed"] is True
    assert cluster["cluster_close_frontier"] == 22000


def test_anticipatory_proposal_fails() -> None:
    with pytest.raises(Phase5PolicyError, match="anticipatory"):
        cluster_proposals(
            [proposal("bad", 20000, 19000, 0.6)],
            cluster_debounce_ms=0,
            cluster_boundary_radius_ms=250,
            refractory_ms=0,
            representative="first",
            episode_observed_end=30000,
        )


def test_cross_profile_replay_fails() -> None:
    left = proposal("p0", 16000, 20000, 0.6)
    right = proposal("p1", 20000, 24000, 0.7)
    right["profile_id"] = "other"
    with pytest.raises(Phase5PolicyError, match="one profile"):
        cluster_proposals(
            [left, right],
            cluster_debounce_ms=0,
            cluster_boundary_radius_ms=250,
            refractory_ms=0,
            representative="first",
            episode_observed_end=30000,
        )


def test_linear_cluster_engine_matches_frozen_reference_grid() -> None:
    rows = [
        proposal("p0", 16000, 20000, 0.6),
        proposal("p1", 16100, 20100, 0.7),
        proposal("p2", 21000, 22000, 0.8),
        proposal("p3", 26000, 27000, 0.5),
        proposal("p4", 26500, 27500, 0.9),
        proposal("p5", 40000, 42000, 0.4),
    ]
    for debounce in (0, 100, 250):
        for radius in (250, 500):
            for refractory in (0, 250, 500):
                for representative in ("first", "max_confidence"):
                    kwargs = {
                        "cluster_debounce_ms": debounce,
                        "cluster_boundary_radius_ms": radius,
                        "refractory_ms": refractory,
                        "representative": representative,
                        "episode_observed_end": 50000,
                    }
                    assert cluster_proposals(rows, **kwargs) == cluster_proposals_reference(
                        rows, **kwargs
                    )


def vad(
    action_id: str,
    boundary: int,
    observed: int,
    *,
    silence: str | None = None,
) -> dict[str, object]:
    return {
        "action_id": action_id,
        "audio_epoch": 7,
        "source_session_id": "source",
        "boundary_source_sample": boundary,
        "observed_source_sample_at_emit": observed,
        "silence_interval_id": silence,
        "action_kind": "retain_vad",
    }


def detector(
    action_id: str,
    boundary: int,
    observed: int,
    *,
    silence: str | None = None,
) -> dict[str, object]:
    return {
        "detector_action_id": action_id,
        "cluster_id": action_id,
        "audio_epoch": 7,
        "source_session_id": "source",
        "proposal_kind": "speaker_change_unknown",
        "requested_action": "hard_candidate",
        "boundary_source_sample": boundary,
        "observed_source_sample_at_emit": observed,
        "confidence": 0.8,
        "confidence_semantics_id": "change.v1",
        "silence_interval_id": silence,
    }


def test_detector_after_vad_is_suppressed() -> None:
    result = fuse_actions(
        [vad("v0", 16000, 17000)],
        [detector("d0", 16200, 18000)],
        detector_vad_radius_ms=250,
        same_silence_interval_association=False,
    )
    assert [row["action_kind"] for row in result["final_actions"]] == ["retain_vad"]
    assert result["suppression_evidence"][0]["action_kind"] == "suppress_detector_duplicate"


def test_vad_after_detector_relabels_acceleration() -> None:
    result = fuse_actions(
        [vad("v0", 16400, 20000)],
        [detector("d0", 16000, 18000)],
        detector_vad_radius_ms=250,
        same_silence_interval_association=False,
    )
    assert result["final_actions"][0]["action_kind"] == "accelerate_or_replace_vad"
    assert result["final_actions"][0]["associated_vad_action_id"] == "v0"
    assert result["suppression_evidence"][0]["action_kind"] == "suppress_vad_duplicate"


def test_same_silence_association_can_exceed_radius() -> None:
    result = fuse_actions(
        [vad("v0", 20000, 22000, silence="s0")],
        [detector("d0", 16000, 18000, silence="s0")],
        detector_vad_radius_ms=250,
        same_silence_interval_association=True,
    )
    assert result["final_actions"][0]["action_kind"] == "accelerate_or_replace_vad"


def test_association_forbidden_preserves_both_actions() -> None:
    later_vad = vad("v0", 16200, 20000)
    later_vad["association_forbidden"] = "ends_new_turn"
    result = fuse_actions(
        [later_vad],
        [detector("d0", 16000, 18000)],
        detector_vad_radius_ms=250,
        same_silence_interval_association=True,
    )
    assert [row["action_kind"] for row in result["final_actions"]] == [
        "add_hard_boundary",
        "retain_vad",
    ]


def lifecycle(event_id: str, kind: str, sample: int, observed: int) -> dict[str, object]:
    return {
        "event_id": event_id,
        "audio_epoch": 7,
        "source_session_id": "source",
        "normalized_utterance_id": event_id,
        "event_kind": kind,
        "reason": "silence",
        "event_source_sample": sample,
        "observed_source_sample_at_emit": observed,
    }


def test_full_replay_derives_same_silence_association_beyond_radius() -> None:
    proposal_rows = [proposal("p0", 16000, 18000, 0.8)]
    vad_rows = [vad("v0", 24000, 24512)]
    lifecycle_rows = [
        lifecycle("end-a", "speech_end", 16000, 22000),
        lifecycle("start-b", "speech_start", 24000, 24512),
    ]
    disabled = full_fusion_replay(
        proposal_rows,
        vad_rows,
        cluster_debounce_ms=0,
        cluster_boundary_radius_ms=250,
        refractory_ms=0,
        representative="first",
        detector_vad_radius_ms=250,
        same_silence_interval_association=False,
        episode_observed_end=30000,
        lifecycle_events=lifecycle_rows,
    )
    enabled = full_fusion_replay(
        proposal_rows,
        vad_rows,
        cluster_debounce_ms=0,
        cluster_boundary_radius_ms=250,
        refractory_ms=0,
        representative="first",
        detector_vad_radius_ms=250,
        same_silence_interval_association=True,
        episode_observed_end=30000,
        lifecycle_events=lifecycle_rows,
    )
    assert disabled["fusion"]["hard_action_count"] == 2
    assert enabled["fusion"]["hard_action_count"] == 1
    assert (
        enabled["fusion"]["suppression_evidence"][0]["association_basis"] == "same_silence_interval"
    )


def test_full_replay_uses_pinned_b0_silence_projection() -> None:
    vad_row = vad("v0", 24000, 24512)
    vad_row["debug"] = {
        "prev_speech_end_sample": 16000,
        "prev_end_reason": "silence",
    }
    result = full_fusion_replay(
        [proposal("p0", 16000, 18000, 0.8)],
        [vad_row],
        cluster_debounce_ms=0,
        cluster_boundary_radius_ms=250,
        refractory_ms=0,
        representative="first",
        detector_vad_radius_ms=250,
        same_silence_interval_association=True,
        episode_observed_end=30000,
        lifecycle_events=[],
    )
    assert result["fusion"]["hard_action_count"] == 1
    assert result["silence_intervals"][0]["projection_source"] == "pinned_b0_vad_debug"


def test_full_replay_derives_new_turn_non_association() -> None:
    result = full_fusion_replay(
        [proposal("p0", 16000, 18000, 0.8)],
        [vad("v0", 19000, 19512)],
        cluster_debounce_ms=0,
        cluster_boundary_radius_ms=250,
        refractory_ms=0,
        representative="first",
        detector_vad_radius_ms=250,
        same_silence_interval_association=True,
        episode_observed_end=30000,
        lifecycle_events=[
            lifecycle("start-a", "speech_start", 0, 512),
            lifecycle("end-a", "speech_end", 17000, 18500),
            lifecycle("start-b", "speech_start", 19000, 19512),
        ],
    )
    assert result["fusion"]["hard_action_count"] == 2
    rejected = [
        row
        for row in result["fusion"]["suppression_evidence"]
        if row["action_kind"] == "association_rejected"
    ]
    assert rejected[0]["reason"] == "vad_ends_post_detector_turn"


def test_one_detector_does_not_absorb_two_vad_boundaries() -> None:
    result = fuse_actions(
        [vad("v0", 16200, 20000), vad("v1", 16400, 21000)],
        [detector("d0", 16000, 18000)],
        detector_vad_radius_ms=250,
        same_silence_interval_association=False,
    )
    assert [row["action_kind"] for row in result["final_actions"]] == [
        "accelerate_or_replace_vad",
        "retain_vad",
    ]


def active(start: int, end: int) -> dict[str, int]:
    return {
        "start": start,
        "end": end,
        "start_observed_source_sample": start + 512,
        "end_observed_source_sample": end + 512,
    }


def test_controls_match_detector_hard_count_and_availability() -> None:
    neural = [
        {
            **detector("d0", 16000, 24000),
            "origin": "detector",
            "event_id": "d0",
            "final_action_id": "final:d0",
            "action_kind": "add_hard_boundary",
        },
        {
            **detector("d1", 32000, 40000),
            "origin": "detector",
            "event_id": "d1",
            "final_action_id": "final:d1",
            "action_kind": "accelerate_or_replace_vad",
        },
    ]
    intervals = [active(0, 48000)]
    energy = [
        {
            "candidate_id": "e0",
            "boundary_source_sample": 8192,
            "observed_source_sample": 10000,
            "change_strength": 0.7,
        },
        {
            "candidate_id": "e1",
            "boundary_source_sample": 24576,
            "observed_source_sample": 30000,
            "change_strength": 0.9,
        },
    ]
    for kind in (
        "uniform_vad_active",
        "causal_energy_change_peak",
        "within_vad_active_position_shuffle",
    ):
        result = frequency_matched_control(
            kind,
            neural,
            intervals,
            energy_candidates=energy,
            forbidden_boundaries=[],
            seed_material="seed",
        )
        assert result["status"] == "complete"
        assert result["placed_hard_action_count"] == 2
        assert [row["observed_source_sample_at_emit"] for row in result["actions"]] == [
            24000,
            40000,
        ]
        assert all(
            row["boundary_source_sample"] <= row["observed_source_sample_at_emit"]
            for row in result["actions"]
        )


def test_control_infeasibility_is_visible() -> None:
    neural = [
        {
            **detector("d0", 16000, 18000),
            "origin": "detector",
            "event_id": "d0",
            "final_action_id": "final:d0",
            "action_kind": "add_hard_boundary",
        }
    ]
    result = frequency_matched_control(
        "uniform_vad_active",
        neural,
        [],
        energy_candidates=[],
        forbidden_boundaries=[],
        seed_material="seed",
    )
    assert result["status"] == "infeasible"
    assert result["placed_hard_action_count"] == 0
    assert result["infeasible_placements"][0]["source_action_id"] == "final:d0"


def test_active_interval_with_anticipatory_evidence_fails() -> None:
    row = active(16000, 32000)
    row["start_observed_source_sample"] = 15000
    with pytest.raises(Phase5PolicyError, match="anticipatory"):
        frequency_matched_control(
            "uniform_vad_active",
            [],
            [row],
            energy_candidates=[],
            forbidden_boundaries=[],
            seed_material="seed",
        )


def test_cluster_progress_holds_safe_frontier_until_emit() -> None:
    clustered = cluster_proposals(
        [proposal("p0", 16000, 20000, 0.8)],
        cluster_debounce_ms=250,
        cluster_boundary_radius_ms=250,
        refractory_ms=0,
        representative="first",
        episode_observed_end=30000,
    )
    rows = policy_progress(
        [
            {
                "observed_source_sample": 20000,
                "safe_boundary_frontier_sample": 16000,
            },
            {
                "observed_source_sample": 30000,
                "safe_boundary_frontier_sample": 30000,
            },
        ],
        clustered["clusters"],
        episode_observed_end=30000,
    )
    by_observed = {row["observed_source_sample"]: row for row in rows}
    assert by_observed[20000]["safe_boundary_frontier_sample"] == 15999
    assert by_observed[24000]["safe_boundary_frontier_sample"] == 16000
    assert by_observed[30000]["safe_boundary_frontier_sample"] == 30000
