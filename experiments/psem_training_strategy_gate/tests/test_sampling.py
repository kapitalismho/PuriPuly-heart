from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.psem_training_strategy_gate import sampling
from experiments.psem_training_strategy_gate.data.label_contract import (
    CanonicalInterval,
    generate_labels,
    load_contract,
)
from experiments.psem_training_strategy_gate.preflight import canonical_sha256
from experiments.psem_training_strategy_gate.sampling import (
    HARD_NEGATIVE_FAMILIES,
    HARD_NEGATIVE_TOPOLOGY_FAMILY,
    POSITIVE_FAMILIES,
    POSITIVE_TOPOLOGY_FAMILY,
    SAMPLING_COUNTS,
    WINDOWS_PER_EPOCH,
    BatchValidityAccumulator,
    CandidateCenter,
    RuntimeSession,
    SamplingContractError,
    epoch_plan,
    load_waveform_window,
    role_by_source,
    target_for_row,
)
from experiments.psem_training_strategy_gate.targets import build_window_targets


def _pools():
    families = (*POSITIVE_FAMILIES, *HARD_NEGATIVE_FAMILIES, "source_time_uniform")
    return {
        family: tuple(
            CandidateCenter(f"source-{index % 3}", 32000 + index * 1600, family)
            for index in range(17)
        )
        for family in families
    }


def test_epoch_plan_has_exact_frozen_mixture_and_balanced_families() -> None:
    plan = epoch_plan(_pools(), 1)
    roles = Counter(role for role, _ in plan)
    families = Counter(center.family for _, center in plan)
    assert len(plan) == WINDOWS_PER_EPOCH
    assert roles == Counter(SAMPLING_COUNTS)
    assert (
        max(families[family] for family in POSITIVE_FAMILIES)
        - min(families[family] for family in POSITIVE_FAMILIES)
        <= 1
    )
    assert (
        max(families[family] for family in HARD_NEGATIVE_FAMILIES)
        - min(families[family] for family in HARD_NEGATIVE_FAMILIES)
        <= 1
    )


def test_epoch_plan_is_deterministic_and_rotates_centers() -> None:
    pools = _pools()
    first = epoch_plan(pools, 1)
    assert first == epoch_plan(pools, 1)
    assert first != epoch_plan(pools, 2)


def test_frozen_split_assigns_every_source_once_without_role_overlap() -> None:
    roles = role_by_source()
    assert len(roles) == 93
    assert Counter(roles.values()) == {
        "PSEM-STRATEGY-TRAIN": 64,
        "PSEM-STRATEGY-DEV": 10,
        "PSEM-STRATEGY-EVAL": 19,
    }


def test_training_family_mapping_keeps_valid_micro_topologies_in_semantic_strata() -> None:
    assert POSITIVE_TOPOLOGY_FAMILY["micro_gap_different_speaker_handoff"] == (
        "silence_gap_different_speaker_handoff"
    )
    assert POSITIVE_TOPOLOGY_FAMILY["micro_overlap_takeover"] == "overlap_takeover"
    assert HARD_NEGATIVE_TOPOLOGY_FAMILY["micro_gap_same_speaker_resume"] == (
        "same_speaker_silence_gap_resume"
    )
    assert HARD_NEGATIVE_TOPOLOGY_FAMILY["micro_overlap_return"] == "overlap_return"


def _bound_session_and_row():
    labels = generate_labels(
        (
            CanonicalInterval(0, 35137, ("A",)),
            CanonicalInterval(35137, 80000, ("B",)),
        ),
        contract=load_contract(version="psem-handoff-v1"),
        scored_start_sample=0,
        scored_end_sample=80000,
    )
    session = RuntimeSession(
        source_id="session",
        role="PSEM-STRATEGY-TRAIN",
        audio_ref="session.wav",
        waveform_sha256="a" * 64,
        labels=labels,
    )
    target = build_window_targets("session", labels, 35200)
    row = {
        "source_id": "session",
        "source_waveform_sha256": "a" * 64,
        "boundary_sample": 35200,
        "window_start_sample": target.window_start_sample,
        "window_end_sample": target.window_end_sample,
        "observed_frontier_sample": target.observed_frontier_sample,
        "target_sha256": canonical_sha256(target.to_dict()),
        "unsnapped_handoff_event_samples": list(target.handoff_event_samples),
    }
    return session, row


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_id", "other"),
        ("source_waveform_sha256", "b" * 64),
        ("boundary_sample", True),
        ("window_start_sample", 3201),
        ("window_end_sample", 51199),
        ("observed_frontier_sample", 51199),
        ("target_sha256", "0" * 64),
        ("unsnapped_handoff_event_samples", []),
    ),
)
def test_target_for_row_rejects_every_stale_window_binding(field, value) -> None:
    session, row = _bound_session_and_row()
    row[field] = value
    with pytest.raises(SamplingContractError):
        target_for_row(row, session)


def test_waveform_loader_uses_recomputed_bound_window(monkeypatch, tmp_path: Path) -> None:
    session, row = _bound_session_and_row()
    calls = []

    def fake_load(path, *, frame_offset, num_frames):
        calls.append((path, frame_offset, num_frames))
        return torch.zeros(1, 48000), 16000

    monkeypatch.setattr(sampling.torchaudio, "load", fake_load)
    waveform = load_waveform_window(row, session, tmp_path)
    assert waveform.shape == (48000,)
    assert calls == [(tmp_path / "session.wav", 3200, 48000)]


def test_batch_validity_accumulator_proves_each_enabled_loss_per_official_batch() -> None:
    session, row = _bound_session_and_row()
    target = target_for_row(row, session)
    accumulator = BatchValidityAccumulator(4)
    for _ in range(4):
        accumulator.add(target)
    minimum = accumulator.finish()
    assert minimum["handoff"] == 4
    assert minimum["state"] == 120
    assert minimum["relation"] > 0

    unsupported = replace(
        target,
        handoff_mask=False,
        state_mask=(False,) * 30,
        relation_pairs=(),
    )
    unsupported_accumulator = BatchValidityAccumulator(4)
    for _ in range(4):
        unsupported_accumulator.add(unsupported)
    assert unsupported_accumulator.finish() == {"handoff": 0, "relation": 0, "state": 0}


def test_batch_validity_accumulator_rejects_partial_batches() -> None:
    session, row = _bound_session_and_row()
    accumulator = BatchValidityAccumulator(4)
    accumulator.add(target_for_row(row, session))
    with pytest.raises(SamplingContractError, match="complete official batches"):
        accumulator.finish()
