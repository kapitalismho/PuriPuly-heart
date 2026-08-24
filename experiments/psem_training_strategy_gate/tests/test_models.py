from __future__ import annotations

from pathlib import Path

import torch

from experiments.psem_training_strategy_gate.data.label_contract import (
    CanonicalInterval,
    generate_labels,
    load_contract,
)
from experiments.psem_training_strategy_gate.losses import (
    LossWeights,
    collate_targets,
    compute_losses,
)
from experiments.psem_training_strategy_gate.models import (
    CELL_DIMENSION,
    SCRATCH_DILATIONS,
    build_model,
    optimizer_groups,
    parameter_inventory,
    wavlm_parameter_allowed,
)
from experiments.psem_training_strategy_gate.targets import build_window_targets


def _takeover_target():
    labels = generate_labels(
        (
            CanonicalInterval(0, 32000, ("A",)),
            CanonicalInterval(32000, 35200, ("A", "B")),
            CanonicalInterval(35200, 80000, ("B",)),
        ),
        contract=load_contract(version="psem-handoff-v1"),
        scored_start_sample=0,
        scored_end_sample=80000,
    )
    return build_window_targets("session", labels, 35200)


def test_scratch_graph_has_exact_geometry_and_parameter_range() -> None:
    model = build_model("SCRATCH-PSEM", cache_root=Path("unused"), seed=7301)
    outputs = model(torch.zeros(2, 48000))
    inventory = parameter_inventory(model)
    assert outputs["cells"].shape == (2, 30, CELL_DIMENSION)
    assert outputs["handoff_logits"].shape == (2,)
    assert outputs["state_logits"].shape == (2, 30, 3)
    assert 5_000_000 <= inventory["total_parameters"] <= 10_000_000
    assert inventory["total_parameters"] == 6_137_797
    assert len(model.encoder.blocks) == len(SCRATCH_DILATIONS) == 8
    assert not any("wavlm" in row["name"].lower() for row in inventory["parameters"])


def test_common_head_initialization_is_arm_independent_for_same_seed() -> None:
    left = build_model("SCRATCH-PSEM", cache_root=Path("unused"), seed=7301)
    right = build_model("SCRATCH-PSEM", cache_root=Path("unused"), seed=7301)
    different = build_model("SCRATCH-PSEM", cache_root=Path("unused"), seed=7302)
    left_state = left.head.state_dict()
    right_state = right.head.state_dict()
    different_state = different.head.state_dict()
    assert all(torch.equal(left_state[name], right_state[name]) for name in left_state)
    assert any(not torch.equal(left_state[name], different_state[name]) for name in left_state)


def test_scratch_optimizer_groups_cover_trainable_parameters_once() -> None:
    model = build_model("SCRATCH-PSEM", cache_root=Path("unused"), seed=7301)
    groups = optimizer_groups(model)
    assigned = [name for group in groups for name in group.parameter_names]
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert [group.name for group in groups] == [
        "common_head_and_projection",
        "scratch_encoder",
    ]
    assert [group.learning_rate for group in groups] == [1e-3, 3e-4]
    assert len(assigned) == len(set(assigned))
    assert sorted(assigned) == sorted(trainable)


def test_finetune_whitelist_is_exact_by_parameter_name() -> None:
    assert wavlm_parameter_allowed("encoder.wavlm.encoder.layers.8.attention.q_proj.weight")
    assert wavlm_parameter_allowed("encoder.wavlm.encoder.layers.11.final_layer_norm.bias")
    assert wavlm_parameter_allowed("encoder.wavlm.encoder.layer_norm.weight")
    assert not wavlm_parameter_allowed("encoder.wavlm.encoder.layers.7.attention.q_proj.weight")
    assert not wavlm_parameter_allowed("encoder.wavlm.feature_projection.projection.weight")
    assert not wavlm_parameter_allowed("encoder.wavlm.feature_extractor.conv_layers.0.conv.weight")


def test_shared_loss_uses_direct_state_and_relation_paths() -> None:
    model = build_model("SCRATCH-PSEM", cache_root=Path("unused"), seed=7301)
    target = _takeover_target()
    outputs = model(torch.randn(1, 48000) * 0.01)
    losses = compute_losses(
        model,
        outputs,
        collate_targets((target,)),
        LossWeights(3.0, (1.0, 1.0, 1.0), (1.0, 1.0)),
    )
    assert losses["handoff_valid_count"] == 1
    assert losses["state_valid_count"] == 30
    assert losses["relation_valid_count"] > 0
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    for prefix in (
        "head.handoff_head.",
        "head.state_head.",
        "head.relation_head.",
        "head.temporal.",
        "encoder.",
    ):
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith(prefix)
        ]
        assert gradients
        assert any(
            gradient is not None and torch.isfinite(gradient).all() and float(gradient.norm()) > 0.0
            for gradient in gradients
        )
