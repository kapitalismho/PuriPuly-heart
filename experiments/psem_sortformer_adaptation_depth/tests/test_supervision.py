from __future__ import annotations

import pytest
import torch

from experiments.psem_sortformer_adaptation_depth.nemo_adapter import (
    SortformerEvidence,
    compact_valid_frames,
)
from experiments.psem_sortformer_adaptation_depth.predictions import prediction_rows
from experiments.psem_sortformer_adaptation_depth.supervision import (
    FRAME_COUNT,
    WARMUP_FRAME_COUNT,
    anchor_timeline,
    build_frame_supervision,
    oracle_anchor_one_hot,
    oracle_mapping_from_frames,
)
from experiments.psem_training_strategy_gate.data.label_contract import (
    CanonicalInterval,
    LabelResult,
)


def _labels() -> LabelResult:
    intervals = (
        CanonicalInterval(0, 240000, ("anchor",)),
        CanonicalInterval(240000, 320000, ("other",), ambiguous=True),
        CanonicalInterval(320000, 480000, ("other",)),
    )
    return LabelResult(
        contract_version="psem-handoff-v1",
        contract_document_sha256="a" * 64,
        sample_rate_hz=16000,
        intervals=intervals,
        activity_labels=(
            {"mask_state": "valid", "state": "singleton"},
            {"mask_state": "masked", "state": "singleton"},
            {"mask_state": "valid", "state": "singleton"},
        ),
        transitions=(),
        topology_episodes=(),
        exposure={},
    )


def test_frame_supervision_excludes_warmup_and_ambiguous_native_frames() -> None:
    episodes = tuple("episode-1" for _ in range(FRAME_COUNT))
    anchors = tuple("anchor" for _ in range(FRAME_COUNT))
    supervision = build_frame_supervision(_labels(), 0, episodes, anchors)
    assert not bool(supervision.psem_mask[:WARMUP_FRAME_COUNT].any())
    assert not bool(supervision.native_mask[:WARMUP_FRAME_COUNT].any())
    assert bool(supervision.anchor_targets[WARMUP_FRAME_COUNT:180].all())
    assert not bool(supervision.native_mask[188:250].any())
    assert bool(supervision.replacement_targets[250:].all())
    assert supervision.arrival_order_speakers == ("anchor", "other")


def test_anchor_timeline_uses_the_fixed_simple_anchor_lifecycle() -> None:
    episode_ids, speakers = anchor_timeline("source", _labels(), 0)
    assert episode_ids[0] is None
    assert speakers[0] is None
    assert episode_ids[3] == "source:A00001"
    assert speakers[3] == "anchor"


def test_oracle_mapping_is_recomputed_from_episode_anchor_support() -> None:
    episodes = tuple("episode-1" for _ in range(FRAME_COUNT))
    anchors = tuple("anchor" for _ in range(FRAME_COUNT))
    supervision = build_frame_supervision(_labels(), 0, episodes, anchors)
    probabilities = torch.zeros((FRAME_COUNT, 4))
    probabilities[:, 2] = 0.9
    alive = torch.ones_like(probabilities)
    assert oracle_mapping_from_frames(probabilities, alive, supervision) == {"episode-1": 2}


def test_oracle_mapping_requires_one_valid_slot_per_episode() -> None:
    encoded = oracle_anchor_one_hot(("a", "b"), {"a": 1, "b": 3})
    assert encoded.tolist() == [[0, 1, 0, 0], [0, 0, 0, 1]]
    with pytest.raises(Exception, match="absent or invalid"):
        oracle_anchor_one_hot(("missing",), {})


def test_native_loss_compaction_removes_masked_frames_before_arrival_mapping() -> None:
    probabilities = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    targets = torch.flip(probabilities, dims=(1,))
    mask = torch.tensor([[True, False, True, False, False, True]])
    compact_probabilities, compact_targets, lengths = compact_valid_frames(
        probabilities, targets, mask
    )
    assert lengths.tolist() == [3]
    assert torch.equal(compact_probabilities, probabilities[:, [0, 2, 5]])
    assert torch.equal(compact_targets, targets[:, [0, 2, 5]])


def test_prediction_rows_retain_source_and_evidence_coordinates_and_raw_logits() -> None:
    evidence = SortformerEvidence(
        probabilities=torch.full((1, 2, 4), 0.5),
        activity_logits=torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
        final_temporal_hidden=torch.zeros((1, 2, 192)),
        slot_alive=torch.ones((1, 2, 4)),
        state_reset=torch.tensor([[[1.0], [0.0]]]),
        evidence_delay_seconds=torch.full((1, 2, 1), 1.04),
    )
    outputs = {
        "anchor_present": torch.tensor([[0.25, 0.5]]),
        "replacement_evidence": torch.tensor([[-0.25, -0.5]]),
    }
    rows = prediction_rows("source", 2560, evidence, outputs, ("e1", "e1"), (0, 1))
    assert rows[0]["source_frame_start_sample"] == 2560
    assert rows[0]["source_frame_end_sample"] == 3840
    assert rows[0]["model_evidence_frontier_source_sample"] == 19200
    assert rows[0]["raw_sortformer_activity_logits"] == [0.0, 1.0, 2.0, 3.0]
    assert rows[1]["oracle_anchor_slot"] == 1


@pytest.mark.parametrize(
    "mutation",
    (
        "nonfinite_logit",
        "nonfinite_psem_logit",
        "nonbinary_alive",
        "dead_stable_slot",
        "altered_delay",
    ),
)
def test_prediction_rows_reject_invalid_runtime_semantics(mutation: str) -> None:
    evidence = SortformerEvidence(
        probabilities=torch.full((1, 1, 4), 0.5),
        activity_logits=torch.zeros((1, 1, 4)),
        final_temporal_hidden=torch.zeros((1, 1, 192)),
        slot_alive=torch.ones((1, 1, 4)),
        state_reset=torch.ones((1, 1, 1)),
        evidence_delay_seconds=torch.full((1, 1, 1), 1.04),
    )
    if mutation == "nonfinite_logit":
        evidence.activity_logits[0, 0, 0] = torch.nan
    outputs = {
        "anchor_present": torch.zeros((1, 1)),
        "replacement_evidence": torch.zeros((1, 1)),
    }
    if mutation == "nonfinite_psem_logit":
        outputs["replacement_evidence"][0, 0] = torch.nan
    elif mutation == "nonbinary_alive":
        evidence.slot_alive[0, 0, 0] = 0.5
    elif mutation == "dead_stable_slot":
        evidence.slot_alive[0, 0, 0] = 0
    elif mutation == "altered_delay":
        evidence.evidence_delay_seconds[0, 0, 0] = 1.05
    with pytest.raises(Exception):
        prediction_rows(
            "source",
            0,
            evidence,
            outputs,
            ("episode",),
            (0,),
        )
