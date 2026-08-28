from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import torch

from experiments.psem_sortformer_adaptation_depth import sampling
from experiments.psem_sortformer_adaptation_depth.sampling import (
    FRAME_SAMPLES,
    HARD_NEGATIVE_FAMILIES,
    POSITIVE_FAMILIES,
    ROLE_COUNTS,
    WINDOW_SAMPLES,
    apply_augmentation,
    augmentation_decision,
    candidate_pools,
    epoch_plan,
    materialize_sampling_manifest,
    select_overfit_rows,
    uniform_ranges,
    validate_sampling_manifest,
)
from experiments.psem_training_strategy_gate.data.label_contract import (
    CanonicalInterval,
    LabelResult,
)
from experiments.psem_training_strategy_gate.sampling import TRAIN_ROLE, RuntimeSession


def _session() -> RuntimeSession:
    end = 30 * 60 * 16000
    intervals = (
        CanonicalInterval(0, end // 2, ("speaker-a",)),
        CanonicalInterval(end // 2, end, ("speaker-a", "speaker-b")),
    )
    activity = (
        {"mask_state": "valid", "state": "singleton"},
        {"mask_state": "valid", "state": "overlap"},
    )
    topologies = [
        ("clean_direct_different_speaker_handoff", 1),
        ("silence_gap_different_speaker_handoff", 1),
        ("overlap_takeover", 1),
        ("same_speaker_silence_gap_resume", 0),
        ("overlap_return", 0),
    ]
    transitions = tuple(
        {
            "mask_state": "valid",
            "primary_topology": topology,
            "handoff_confirmed": target,
            "handoff_source_sample": (index + 4) * 16000 * 60,
        }
        for index, (topology, target) in enumerate(topologies)
    )
    labels = LabelResult(
        contract_version="psem-handoff-v1",
        contract_document_sha256="a" * 64,
        sample_rate_hz=16000,
        intervals=intervals,
        activity_labels=activity,
        transitions=transitions,
        topology_episodes=(),
        exposure={},
    )
    return RuntimeSession(
        source_id="train-source",
        role=TRAIN_ROLE,
        audio_ref="audio.wav",
        waveform_sha256="b" * 64,
        labels=labels,
    )


def test_epoch_plan_is_deterministic_and_has_exact_mixture() -> None:
    sessions = {"train-source": _session()}
    pools = candidate_pools(sessions)
    ranges = uniform_ranges(sessions)
    first = epoch_plan(pools, ranges, 1)
    second = epoch_plan(pools, ranges, 1)
    assert first == second
    assert Counter(role for role, _ in first) == Counter(ROLE_COUNTS)
    assert {candidate.family for role, candidate in first if role == "replacement_positive"} == set(
        POSITIVE_FAMILIES
    )
    assert {candidate.family for role, candidate in first if role == "hard_negative"} == set(
        HARD_NEGATIVE_FAMILIES
    )
    assert all(
        candidate.window_end_sample - candidate.window_start_sample == WINDOW_SAMPLES
        for _, candidate in first
    )
    assert all(candidate.window_start_sample % FRAME_SAMPLES == 0 for _, candidate in first)


def test_uniform_windows_are_unique_across_the_full_training_budget() -> None:
    sessions = {"train-source": _session()}
    pools = candidate_pools(sessions)
    ranges = uniform_ranges(sessions)
    windows = [
        candidate.identity
        for epoch in range(1, 9)
        for role, candidate in epoch_plan(pools, ranges, epoch)
        if role == "source_time_uniform"
    ]
    assert len(windows) == len(set(windows)) == 8 * ROLE_COUNTS["source_time_uniform"]


def test_augmentation_is_label_independent_and_uses_only_authorized_families() -> None:
    decision = augmentation_decision("epoch-01-window-0000")
    assert set(decision) == {
        "recipe_version",
        "decision_key",
        "global_gain",
        "additive_non_speech_noise",
        "light_reverberation",
        "band_limitation",
    }
    assert decision == augmentation_decision("epoch-01-window-0000")
    waveform = torch.linspace(-0.5, 0.5, WINDOW_SAMPLES)
    assert apply_augmentation(waveform, decision).shape == waveform.shape


def test_overfit_subset_hash_rule_selects_two_sources_and_fifteen_windows_per_corpus() -> None:
    corpus_by_source = {
        **{f"ami-{index}": "AMI" for index in range(3)},
        **{f"ali-{index}": "AliMeeting" for index in range(3)},
    }
    rows = [
        {
            "source_id": source_id,
            "split_role": TRAIN_ROLE,
            "window_start_sample": window * WINDOW_SAMPLES,
            "window_end_sample": (window + 1) * WINDOW_SAMPLES,
        }
        for source_id in corpus_by_source
        for window in range(20)
    ]
    selected = select_overfit_rows(rows, corpus_by_source)
    counts = Counter(row["source_id"] for row in selected)
    assert len(counts) == 4
    assert set(counts.values()) == {15}
    assert Counter(corpus_by_source[source] for source in counts) == Counter(
        {"AMI": 2, "AliMeeting": 2}
    )


def test_materialized_manifest_round_trips_through_the_exact_validator(
    tmp_path: Path, monkeypatch
) -> None:
    sessions = {"train-source": _session()}
    monkeypatch.setattr(
        sampling,
        "_train_split_binding",
        lambda _: {
            "data_split_receipt_sha256": "c" * 64,
            "split_manifest_sha256": "d" * 64,
            "source_manifest_sha256": "e" * 64,
            "train_source_count": 1,
        },
    )
    monkeypatch.setattr(
        sampling,
        "_source_rows",
        lambda: {
            "train-source": {
                "source_id": "train-source",
                "corpus": "AMI",
                "audio_ref": "audio.wav",
                "waveform_sha256": "b" * 64,
            }
        },
    )
    path = tmp_path / "sampling.jsonl"
    receipt = materialize_sampling_manifest(sessions, path)
    validated = validate_sampling_manifest(path, sessions)
    assert receipt["row_count"] == 8 * 4096
    assert validated["passed"]
    assert validated["manifest_sha256"] == receipt["manifest_sha256"]


def test_official_manifest_rejects_a_relabelled_or_partial_train_split() -> None:
    with pytest.raises(Exception, match="exact frozen TRAIN split"):
        sampling._train_split_binding({"train-source": _session()})
