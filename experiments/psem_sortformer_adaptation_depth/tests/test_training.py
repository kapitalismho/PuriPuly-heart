from __future__ import annotations

import copy
import json
import pickle
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from experiments.psem_sortformer_adaptation_depth.nemo_adapter import SortformerEvidence
from experiments.psem_sortformer_adaptation_depth.preflight import canonical_sha256
from experiments.psem_sortformer_adaptation_depth.supervision import FRAME_COUNT, FrameSupervision
from experiments.psem_sortformer_adaptation_depth.training import (
    _TRAINING_EXAMPLE_TOKEN,
    EFFECTIVE_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    MICRO_BATCH_SIZE,
    OFFICIAL_OPTIMIZER_STEPS,
    SMOKE_OPTIMIZER_STEPS,
    TrainingExample,
    _batch_supervision,
    _training_example_content_bound,
    authorize_official_training,
    build_manifest_class_weight_receipt,
    derive_train_class_weights,
    duration_weighted_average_precision,
    warmup_scheduler,
)


def _example() -> TrainingExample:
    anchor = torch.tensor([0.0, 1.0, 1.0, 0.0])
    replacement = torch.tensor([1.0, 0.0, 0.0, 1.0])
    supervision = FrameSupervision(
        anchor_targets=anchor,
        replacement_targets=replacement,
        psem_mask=torch.ones(4),
        arrival_order_targets=torch.zeros((4, 4)),
        native_mask=torch.ones(4, dtype=torch.bool),
        arrival_order_speakers=("a", "b"),
        mapping_anchor_active=anchor.bool(),
        anchor_episode_ids=("e", "e", "e", "e"),
    )
    return TrainingExample("source", "AMI", "row", torch.zeros(480000), supervision)


def test_class_weights_are_derived_only_from_active_train_targets() -> None:
    weights = derive_train_class_weights([_example()])
    assert weights.replacement_positive == 1
    assert weights.anchor_positive == 1


def test_manifest_class_weight_receipt_survives_json_round_trip(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "sampling.jsonl"
    row = {
        "row_id": "epoch-01-window-0000",
        "split_role": "PSEM-STRATEGY-TRAIN",
        "source_id": "source",
        "window_start_sample": 0,
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    supervision = _example().supervision
    monkeypatch.setattr(
        "experiments.psem_sortformer_adaptation_depth.training.validate_sampling_manifest",
        lambda *_args: {"passed": True},
    )
    monkeypatch.setattr(
        "experiments.psem_sortformer_adaptation_depth.training.anchor_timeline",
        lambda *_args: (("e",), ("speaker",)),
    )
    monkeypatch.setattr(
        "experiments.psem_sortformer_adaptation_depth.training.build_frame_supervision",
        lambda *_args: supervision,
    )
    receipt = build_manifest_class_weight_receipt(
        [row],
        {"source": SimpleNamespace(role="PSEM-STRATEGY-TRAIN", labels=object())},
        manifest,
    )
    assert json.loads(json.dumps(receipt)) == receipt


def test_official_example_attestation_requires_the_factory_token() -> None:
    example = _example()
    sealed = replace(
        example,
        _factory_token=_TRAINING_EXAMPLE_TOKEN,
    )
    assert _training_example_content_bound(sealed)
    assert _training_example_content_bound(pickle.loads(pickle.dumps(sealed)))
    assert not _training_example_content_bound(example)


def test_duration_weighted_average_precision_uses_only_unmasked_frames() -> None:
    logits = torch.tensor([[4.0, -4.0, 3.0, -3.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    assert duration_weighted_average_precision(logits, targets, mask) == 1


def test_warmup_uses_the_exact_thirteen_steps_of_the_256_step_recipe() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = warmup_scheduler(optimizer, 256, 13)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1 / 13)
    for _ in range(12):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)


def test_supported_training_budgets_are_the_lean_recipe() -> None:
    assert SMOKE_OPTIMIZER_STEPS == 32
    assert OFFICIAL_OPTIMIZER_STEPS == 256
    assert MICRO_BATCH_SIZE == 2
    assert GRADIENT_ACCUMULATION_STEPS == 8
    assert EFFECTIVE_BATCH_SIZE == 16


def test_anchor_free_uniform_window_is_safely_masked_before_oracle_mapping() -> None:
    supervision = FrameSupervision(
        anchor_targets=torch.zeros(FRAME_COUNT),
        replacement_targets=torch.zeros(FRAME_COUNT),
        psem_mask=torch.zeros(FRAME_COUNT),
        arrival_order_targets=torch.zeros((FRAME_COUNT, 4)),
        native_mask=torch.ones(FRAME_COUNT, dtype=torch.bool),
        arrival_order_speakers=("speaker",),
        mapping_anchor_active=torch.zeros(FRAME_COUNT, dtype=torch.bool),
        anchor_episode_ids=tuple(None for _ in range(FRAME_COUNT)),
    )
    example = TrainingExample("source", "AMI", "row", torch.zeros(480000), supervision)
    evidence = SortformerEvidence(
        probabilities=torch.full((1, FRAME_COUNT, 4), 0.25),
        activity_logits=torch.zeros((1, FRAME_COUNT, 4)),
        final_temporal_hidden=torch.zeros((1, FRAME_COUNT, 192)),
        slot_alive=torch.ones((1, FRAME_COUNT, 4), dtype=torch.bool),
        state_reset=torch.zeros((1, FRAME_COUNT, 1)),
        evidence_delay_seconds=torch.full((1, FRAME_COUNT, 1), 1.04),
    )
    anchor_one_hot, _, _, mask, _ = _batch_supervision((example,), evidence)
    assert not bool(mask.any())
    assert bool((anchor_one_hot[..., 0] == 1).all())


def test_episode_without_anchor_active_support_is_excluded_from_psem_loss() -> None:
    split = FRAME_COUNT // 2
    mapping_support = torch.zeros(FRAME_COUNT, dtype=torch.bool)
    mapping_support[0] = True
    supervision = FrameSupervision(
        anchor_targets=torch.zeros(FRAME_COUNT),
        replacement_targets=torch.ones(FRAME_COUNT),
        psem_mask=torch.ones(FRAME_COUNT),
        arrival_order_targets=torch.zeros((FRAME_COUNT, 4)),
        native_mask=torch.ones(FRAME_COUNT, dtype=torch.bool),
        arrival_order_speakers=("speaker",),
        mapping_anchor_active=mapping_support,
        anchor_episode_ids=tuple(
            "mapped" if index < split else "unsupported" for index in range(FRAME_COUNT)
        ),
    )
    example = TrainingExample("source", "AMI", "row", torch.zeros(480000), supervision)
    probabilities = torch.zeros((1, FRAME_COUNT, 4))
    probabilities[:, :, 2] = 1
    evidence = SortformerEvidence(
        probabilities=probabilities,
        activity_logits=torch.zeros((1, FRAME_COUNT, 4)),
        final_temporal_hidden=torch.zeros((1, FRAME_COUNT, 192)),
        slot_alive=torch.ones((1, FRAME_COUNT, 4), dtype=torch.bool),
        state_reset=torch.zeros((1, FRAME_COUNT, 1)),
        evidence_delay_seconds=torch.full((1, FRAME_COUNT, 1), 1.04),
    )
    anchor_one_hot, _, _, mask, _ = _batch_supervision((example,), evidence)
    assert bool(mask[0, :split].all())
    assert not bool(mask[0, split:].any())
    assert bool((anchor_one_hot[0, :split, 2] == 1).all())


def test_official_authorization_binds_every_source_window_target_and_augmentation() -> None:
    rows = [
        {
            "row_id": f"epoch-01-window-{index:04d}",
            "epoch": 1,
            "epoch_index": index,
            "split_role": "PSEM-STRATEGY-TRAIN",
            "source_id": "source",
            "corpus": "AMI",
            "window_start_sample": index * 1280,
            "window_end_sample": index * 1280 + 480000,
            "target_identity_sha256": "a" * 64,
            "augmentation_identity_sha256": "b" * 64,
            "state_reset_at_window_start": True,
        }
        for index in range(4096)
    ]
    weight_payload = {
        "schema_version": 1,
        "artifact_role": "train_class_weight_receipt",
        "sampling_manifest_sha256": "c" * 64,
        "replacement_positive_weight": 2.0,
        "anchor_positive_weight": 3.0,
    }
    weights = {**weight_payload, "payload_sha256": canonical_sha256(weight_payload)}
    shared = canonical_sha256(
        [
            {
                key: row[key]
                for key in (
                    "row_id",
                    "source_id",
                    "corpus",
                    "window_start_sample",
                    "window_end_sample",
                    "target_identity_sha256",
                    "augmentation_identity_sha256",
                    "state_reset_at_window_start",
                )
            }
            for row in rows
        ]
    )
    gate_payload = {
        "schema_version": 1,
        "artifact_role": "material_training_authorization",
        "passed": True,
        "arm": "H-HEAD",
        "seed": 7301,
        "git_head": "a" * 40,
        "sampling_manifest_sha256": "c" * 64,
        "class_weight_receipt_sha256": weights["payload_sha256"],
        "shared_input_identity_sha256": shared,
        "dev_source_ids_sha256": "d" * 64,
    }
    gate = {**gate_payload, "payload_sha256": canonical_sha256(gate_payload)}
    authorization = authorize_official_training(gate, rows, weights)
    assert authorization.arm == "H-HEAD"
    forged = copy.deepcopy(rows)
    forged[0]["source_id"] = "eval-source"
    with pytest.raises(Exception, match="shared targets and augmentations"):
        authorize_official_training(gate, forged, weights)
