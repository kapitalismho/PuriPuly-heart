from __future__ import annotations

import json

import numpy as np

from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
    ACTION_REFERENCE_COVERAGE_PATH,
    ACTION_REFERENCE_LEDGER_PATH,
    MAPPING_COVERAGE_PATH,
    MAPPING_LEDGER_PATH,
    PACKAGE_ROOT,
    SessionExamples,
    config,
    load_sessions,
)
from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import decode_scores
from experiments.psem_frozen_ceiling_gate.posterior_features import (
    TemporalContract,
    temporal_features,
)


def test_causal_features_do_not_use_future_or_cross_episode() -> None:
    contract = TemporalContract((0, 1), (1,))
    episodes = np.asarray(["a", "a", "b", "b"])
    first = np.asarray([[1.0], [2.0], [10.0], [20.0]], dtype=np.float32)
    changed = first.copy()
    changed[1] = 99.0
    causal_first = temporal_features(first, episodes, contract, noncausal=False)
    causal_changed = temporal_features(changed, episodes, contract, noncausal=False)
    assert causal_first[0].tolist() == causal_changed[0].tolist()
    assert causal_first[2, 1] == 10.0
    noncausal_first = temporal_features(first, episodes, contract, noncausal=True)
    noncausal_changed = temporal_features(changed, episodes, contract, noncausal=True)
    assert noncausal_first[0].tolist() != noncausal_changed[0].tolist()


def test_pre_roll_cannot_accumulate_confirmation_and_future_moves_frontier() -> None:
    session = SessionExamples(
        role="eval",
        source_id="source",
        source_family="family",
        confirmation_ms=100,
        manifest={},
        reference=None,
        mapping_records=(),
        episode_ids=np.asarray(["episode", "episode", "episode"]),
        episode_speakers=np.asarray(["A", "A", "A"]),
        starts=np.asarray([0, 1600, 3200]),
        ends=np.asarray([1600, 3200, 4800]),
        frontiers=np.asarray([2000, 3600, 5200]),
        probabilities=np.zeros((3, 4), dtype=np.float32),
        alive=np.ones((3, 4), dtype=np.bool_),
        reset=np.zeros(3, dtype=np.bool_),
        valid=np.ones(3, dtype=np.bool_),
        masked=np.zeros(3, dtype=np.bool_),
        speech_present=np.ones(3, dtype=np.bool_),
        anchor_present=np.zeros(3, dtype=np.bool_),
        overlap=np.zeros(3, dtype=np.bool_),
    )
    events = decode_scores(
        session,
        np.ones(3),
        threshold=0.5,
        confirmation_ms=100,
        future_context_frames=1,
        confirmation_support=[(500, 4800)],
    )
    assert len(events) == 1
    assert events[0].boundary_source_sample == 500
    assert events[0].model_evidence_frontier_sample == 5200
    assert events[0].decoder_emit_sample >= events[0].model_evidence_frontier_sample


def test_split_is_family_level_and_scoring_sources_do_not_train_their_fold() -> None:
    split = json.loads((PACKAGE_ROOT / "split_manifest.json").read_text(encoding="utf-8"))
    source_index = {value["source_id"]: value for value in split["sources"]}
    for fold in split["folds"]:
        assert fold["held_out_family"] not in fold["train_families"]
        assert set(fold["training_sources"]).isdisjoint(fold["evaluation_sources"])
        assert all(
            source_index[source]["source_family"] != fold["held_out_family"]
            and source_index[source]["old_v2_role"] == "PSEM-STRATEGY-DEV"
            for source in fold["training_sources"]
        )
        assert all(
            source_index[source]["source_family"] == fold["held_out_family"]
            and source_index[source]["old_v2_role"] == "PSEM-STRATEGY-EVAL"
            for source in fold["evaluation_sources"]
        )


def test_probe_family_and_noncausal_horizon_are_frozen() -> None:
    cfg = config()
    assert cfg["probe_classes"] == ["linear", "tiny_mlp"]
    assert cfg["noncausal_horizon_ms"] == 500
    assert max(cfg["noncausal_future_frames"]) == 5


def test_oracle_ledger_preserves_unmapped_denominator_and_episode_support() -> None:
    rows = [
        json.loads(value) for value in MAPPING_LEDGER_PATH.read_text(encoding="utf-8").splitlines()
    ]
    coverage = json.loads(MAPPING_COVERAGE_PATH.read_text(encoding="utf-8"))
    keys = {
        (
            value["old_v2_role"],
            value["source_id"],
            value["confirmation_ms"],
            value["anchor_episode_id"],
        )
        for value in rows
    }
    assert len(keys) == len(rows) == coverage["episode_count"]
    assert (
        sum(value["status"] == "unmapped" for value in rows) == coverage["unmapped_episode_count"]
    )
    sessions = load_sessions((500,))
    for session in sessions:
        assert len(session.mapping_records) == len(session.reference.episodes)
        episodes = {value.episode_id: value for value in session.reference.episodes}
        assert all(
            start >= episodes[str(episode_id)].anchor_emit_sample
            and end <= episodes[str(episode_id)].end_emit_sample
            for episode_id, start, end in zip(
                session.episode_ids, session.starts, session.ends, strict=True
            )
        )


def test_action_reference_ledger_is_complete_and_consumed() -> None:
    rows = [
        json.loads(value)
        for value in ACTION_REFERENCE_LEDGER_PATH.read_text(encoding="utf-8").splitlines()
    ]
    coverage = json.loads(ACTION_REFERENCE_COVERAGE_PATH.read_text(encoding="utf-8"))
    assert len(rows) == coverage["episode_count"]
    assert coverage["dev_gate0_exact_event_match"] is True
    assert sum(value["reference_event"] is not None for value in rows) == coverage[
        "reference_event_count"
    ]
    ledger_events = {
        (value["old_v2_role"], value["source_id"], value["anchor_episode_id"]): value[
            "reference_event"
        ]
        for value in rows
        if value["reference_event"] is not None
    }
    sessions = load_sessions((500,))
    consumed_events = {
        (session.manifest["role"], session.source_id, event.anchor_episode_id): event.to_dict()
        for session in sessions
        for event in session.reference.events
    }
    assert consumed_events == ledger_events


def test_final_report_renderer_has_no_placeholder_answers() -> None:
    source = (PACKAGE_ROOT / "evaluate_ceiling.py").read_text(encoding="utf-8")
    assert "Pending scientific interpretation" not in source
    assert "gap_reference_confirmation_ms" in source
