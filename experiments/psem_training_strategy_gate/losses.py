from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from experiments.psem_training_strategy_gate.models import PSEMModel
from experiments.psem_training_strategy_gate.targets import WindowTargets


class LossContractError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LossWeights:
    handoff_positive: float
    state_classes: tuple[float, float, float]
    relation_classes: tuple[float, float]


@dataclass(frozen=True, slots=True)
class TargetBatch:
    handoff_targets: torch.Tensor
    handoff_mask: torch.Tensor
    state_targets: torch.Tensor
    state_mask: torch.Tensor
    relation_batch_indices: torch.Tensor
    relation_left_cells: torch.Tensor
    relation_right_cells: torch.Tensor
    relation_targets: torch.Tensor

    def to(self, device: torch.device | str) -> TargetBatch:
        return TargetBatch(
            *(getattr(self, field).to(device) for field in self.__dataclass_fields__)
        )


@dataclass(slots=True)
class LossAccumulator:
    handoff_numerator: float = 0.0
    handoff_denominator: float = 0.0
    state_numerator: float = 0.0
    state_denominator: float = 0.0
    relation_numerator: float = 0.0
    relation_denominator: float = 0.0

    def update(self, statistics: dict[str, float]) -> None:
        for field in self.__dataclass_fields__:
            value = statistics.get(field)
            if not isinstance(value, float) or not torch.isfinite(torch.tensor(value)):
                raise LossContractError("loss statistics must be finite floats")
            setattr(self, field, getattr(self, field) + value)

    def result(self) -> dict[str, float]:
        denominators = (
            self.handoff_denominator,
            self.state_denominator,
            self.relation_denominator,
        )
        if any(value <= 0.0 for value in denominators):
            raise LossContractError("every enabled loss requires DEV support")
        handoff = self.handoff_numerator / self.handoff_denominator
        state = self.state_numerator / self.state_denominator
        relation = self.relation_numerator / self.relation_denominator
        return {
            "total": handoff + 0.5 * state + 0.5 * relation,
            "handoff": handoff,
            "state": state,
            "relation": relation,
        }


def collate_targets(targets: Sequence[WindowTargets]) -> TargetBatch:
    if not targets:
        raise LossContractError("at least one target window is required")
    relation_batch_indices: list[int] = []
    relation_left_cells: list[int] = []
    relation_right_cells: list[int] = []
    relation_targets: list[float] = []
    for batch_index, target in enumerate(targets):
        for pair in target.relation_pairs:
            relation_batch_indices.append(batch_index)
            relation_left_cells.append(pair.left_cell)
            relation_right_cells.append(pair.right_cell)
            relation_targets.append(float(pair.target))
    return TargetBatch(
        handoff_targets=torch.tensor(
            [target.handoff_target for target in targets], dtype=torch.float32
        ),
        handoff_mask=torch.tensor([target.handoff_mask for target in targets], dtype=torch.bool),
        state_targets=torch.tensor([target.state_targets for target in targets], dtype=torch.long),
        state_mask=torch.tensor([target.state_mask for target in targets], dtype=torch.bool),
        relation_batch_indices=torch.tensor(relation_batch_indices, dtype=torch.long),
        relation_left_cells=torch.tensor(relation_left_cells, dtype=torch.long),
        relation_right_cells=torch.tensor(relation_right_cells, dtype=torch.long),
        relation_targets=torch.tensor(relation_targets, dtype=torch.float32),
    )


def compute_losses(
    model: PSEMModel,
    outputs: dict[str, torch.Tensor],
    targets: TargetBatch,
    weights: LossWeights,
) -> dict[str, torch.Tensor | int]:
    handoff_logits = outputs["handoff_logits"][targets.handoff_mask]
    handoff_targets = targets.handoff_targets[targets.handoff_mask]
    state_logits = outputs["state_logits"][targets.state_mask]
    state_targets = targets.state_targets[targets.state_mask]
    relation_count = int(targets.relation_targets.numel())
    if not handoff_targets.numel() or not state_targets.numel() or not relation_count:
        raise LossContractError("every enabled loss requires at least one valid example")
    handoff_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        handoff_logits,
        handoff_targets,
        pos_weight=torch.tensor(
            weights.handoff_positive,
            device=handoff_logits.device,
            dtype=handoff_logits.dtype,
        ),
    )
    state_loss = torch.nn.functional.cross_entropy(
        state_logits,
        state_targets,
        weight=torch.tensor(
            weights.state_classes,
            device=state_logits.device,
            dtype=state_logits.dtype,
        ),
    )
    relation_logits = model.head.relation_logits(
        outputs["hidden"],
        targets.relation_batch_indices,
        targets.relation_left_cells,
        targets.relation_right_cells,
    )
    relation_class_weights = torch.tensor(
        weights.relation_classes,
        device=relation_logits.device,
        dtype=relation_logits.dtype,
    )
    relation_weights = relation_class_weights[targets.relation_targets.long()]
    relation_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        relation_logits,
        targets.relation_targets,
        weight=relation_weights,
    )
    total = handoff_loss + 0.5 * state_loss + 0.5 * relation_loss
    return {
        "total": total,
        "handoff": handoff_loss,
        "state": state_loss,
        "relation": relation_loss,
        "handoff_valid_count": int(handoff_targets.numel()),
        "state_valid_count": int(state_targets.numel()),
        "relation_valid_count": relation_count,
    }


def loss_statistics(
    model: PSEMModel,
    outputs: dict[str, torch.Tensor],
    targets: TargetBatch,
    weights: LossWeights,
) -> dict[str, float]:
    handoff_logits = outputs["handoff_logits"][targets.handoff_mask]
    handoff_targets = targets.handoff_targets[targets.handoff_mask]
    state_logits = outputs["state_logits"][targets.state_mask]
    state_targets = targets.state_targets[targets.state_mask]
    handoff_numerator = torch.nn.functional.binary_cross_entropy_with_logits(
        handoff_logits,
        handoff_targets,
        pos_weight=torch.tensor(
            weights.handoff_positive,
            device=handoff_logits.device,
            dtype=handoff_logits.dtype,
        ),
        reduction="sum",
    )
    state_weights = torch.tensor(
        weights.state_classes,
        device=state_logits.device,
        dtype=state_logits.dtype,
    )
    state_numerator = torch.nn.functional.cross_entropy(
        state_logits,
        state_targets,
        weight=state_weights,
        reduction="sum",
    )
    relation_count = int(targets.relation_targets.numel())
    if relation_count:
        relation_logits = model.head.relation_logits(
            outputs["hidden"],
            targets.relation_batch_indices,
            targets.relation_left_cells,
            targets.relation_right_cells,
        )
        relation_class_weights = torch.tensor(
            weights.relation_classes,
            device=relation_logits.device,
            dtype=relation_logits.dtype,
        )
        relation_weights = relation_class_weights[targets.relation_targets.long()]
        relation_numerator = torch.nn.functional.binary_cross_entropy_with_logits(
            relation_logits,
            targets.relation_targets,
            weight=relation_weights,
            reduction="sum",
        )
    else:
        relation_numerator = outputs["hidden"].new_zeros(())
    return {
        "handoff_numerator": float(handoff_numerator),
        "handoff_denominator": float(handoff_targets.numel()),
        "state_numerator": float(state_numerator),
        "state_denominator": float(state_weights[state_targets].sum()),
        "relation_numerator": float(relation_numerator),
        "relation_denominator": float(relation_count),
    }
