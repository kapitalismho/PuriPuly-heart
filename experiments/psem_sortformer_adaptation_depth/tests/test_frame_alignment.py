from __future__ import annotations

import numpy as np
import pytest

from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import load_sessions
from experiments.psem_sortformer_adaptation_depth.evaluation import _adapt_session
from experiments.psem_sortformer_adaptation_depth.frame_alignment import (
    action_sample_indices,
    mapping_from_action_probabilities,
    native_episode_timeline,
    native_frame_coordinates,
)


def _native_rows(session) -> list[dict]:
    frame_count = int(session.ends[-1]) // 1280
    starts, ends = native_frame_coordinates(frame_count)
    episodes = native_episode_timeline(session.reference, frame_count)
    probabilities = np.full((len(session.starts), 4), 0.1, dtype=np.float32)
    probabilities[:, 2] = 0.9
    slots, _ = mapping_from_action_probabilities(
        session, probabilities, np.ones_like(probabilities, dtype=np.bool_)
    )
    return [
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_frame_prediction",
            "source_id": session.source_id,
            "source_frame_start_sample": int(start),
            "source_frame_end_sample": int(end),
            "model_evidence_frontier_source_sample": int(start) + 16640,
            "anchor_episode_id": episode,
            "oracle_anchor_slot": slots.get(episode, 0),
            "slot_alive": [True, True, True, True],
            "state_reset": index == 0,
            "raw_sortformer_activity_logits": [-2.0, -2.0, 2.0, -2.0],
            "raw_anchor_present_logit": 1.0,
            "raw_replacement_evidence_logit": -1.0,
        }
        for index, (start, end, episode) in enumerate(zip(starts, ends, episodes, strict=True))
    ]


def test_native_80ms_predictions_align_causally_to_committed_issue_99_action_grid() -> None:
    session = load_sessions(validate_mapping_ledger=False)[0]
    rows = _native_rows(session)
    adapted, selected, logits, diagnostics = _adapt_session(session, rows)
    selected_native_ends = adapted.frontiers - 16640 + 1280
    _, native_ends = native_frame_coordinates(len(rows))
    native_episodes = tuple(row["anchor_episode_id"] for row in rows)
    action_episodes = tuple(
        None if str(value) in {"", "None"} else str(value) for value in session.episode_ids
    )
    indices = action_sample_indices(native_ends, session.ends)
    assert len(rows) != len(session.starts)
    assert np.array_equal(selected_native_ends, native_ends[indices])
    assert np.all(selected_native_ends <= session.ends)
    assert any(
        native_episodes[index] != episode
        for index, episode in zip(indices, action_episodes, strict=True)
    )
    assert selected.shape == (len(session.starts),)
    assert logits.shape == (len(session.starts), 2)
    assert diagnostics["slot_instability_count"] == 0


def test_action_evaluation_rejects_stale_native_oracle_slots() -> None:
    session = load_sessions(validate_mapping_ledger=False)[0]
    rows = _native_rows(session)
    first_episode = next(index for index, row in enumerate(rows) if row["anchor_episode_id"])
    rows[first_episode]["oracle_anchor_slot"] = 1
    with pytest.raises(Exception, match="stale oracle slot mapping"):
        _adapt_session(session, rows)
