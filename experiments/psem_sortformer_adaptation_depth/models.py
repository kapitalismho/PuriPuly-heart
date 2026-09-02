from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

HIDDEN_DIMENSION = 192
SLOT_COUNT = 4
PSEM_INPUT_DIMENSION = 208
MODEL_EVIDENCE_DELAY_SECONDS = 1.04
TRAIN_ROLE = "PSEM-STRATEGY-TRAIN"
DEV_ROLE = "PSEM-STRATEGY-DEV"
NATIVE_SORTFORMER_LOSS_KIND = "checkpoint_arrival_order_four_slot_loss"
NATIVE_SORTFORMER_LOSS_ORIGIN = "nemo.streaming_sortformer.native_loss"
NATIVE_SORTFORMER_CHECKPOINT_SHA256 = (
    "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"
)
_NATIVE_LOSS_TOKEN = object()


@dataclass(frozen=True)
class NativeSortformerLoss:
    value: torch.Tensor
    sampling_roles: tuple[str, ...]
    _token: object


def _require_binary(name: str, value: torch.Tensor) -> None:
    condition = torch.all(torch.logical_or(value == 0, value == 1))
    if condition.device.type == "cuda":
        torch._assert_async(condition, f"{name} must be binary")
    elif not bool(condition):
        raise ValueError(f"{name} must be binary")


def _loss_roles(sampling_roles: Sequence[str]) -> tuple[str, ...]:
    roles = tuple(sampling_roles)
    if not roles or len(set(roles)) != 1 or roles[0] not in {TRAIN_ROLE, DEV_ROLE}:
        raise ValueError("loss inputs must contain one homogeneous TRAIN or DEV role")
    return roles


def bind_native_sortformer_loss(
    value: torch.Tensor,
    *,
    sampling_roles: Sequence[str],
    kind: str,
    origin: str,
    checkpoint_sha256: str,
) -> NativeSortformerLoss:
    roles = _loss_roles(sampling_roles)
    if value.ndim != 0:
        raise ValueError("native Sortformer loss must be one scalar")
    finite = torch.isfinite(value)
    if finite.device.type == "cuda":
        torch._assert_async(finite, "native Sortformer loss must be finite")
    elif not bool(finite):
        raise ValueError("native Sortformer loss must be finite")
    if kind != NATIVE_SORTFORMER_LOSS_KIND:
        raise ValueError("native Sortformer loss kind differs from the frozen contract")
    if origin != NATIVE_SORTFORMER_LOSS_ORIGIN:
        raise ValueError("native Sortformer loss origin differs from the frozen adapter")
    if checkpoint_sha256 != NATIVE_SORTFORMER_CHECKPOINT_SHA256:
        raise ValueError("native Sortformer loss checkpoint differs from the frozen checkpoint")
    return NativeSortformerLoss(value=value, sampling_roles=roles, _token=_NATIVE_LOSS_TOKEN)


def build_psem_features(
    final_temporal_hidden: torch.Tensor,
    speaker_activity_logits: torch.Tensor,
    slot_alive: torch.Tensor,
    oracle_anchor_slot_one_hot: torch.Tensor,
    state_reset: torch.Tensor,
    model_evidence_delay_seconds: torch.Tensor,
) -> torch.Tensor:
    prefix = final_temporal_hidden.shape[:-1]
    expected = {
        "final_temporal_hidden": (*prefix, HIDDEN_DIMENSION),
        "speaker_activity_logits": (*prefix, SLOT_COUNT),
        "slot_alive": (*prefix, SLOT_COUNT),
        "oracle_anchor_slot_one_hot": (*prefix, SLOT_COUNT),
        "state_reset": (*prefix, 1),
        "model_evidence_delay_seconds": (*prefix, 1),
    }
    actual = {
        "final_temporal_hidden": tuple(final_temporal_hidden.shape),
        "speaker_activity_logits": tuple(speaker_activity_logits.shape),
        "slot_alive": tuple(slot_alive.shape),
        "oracle_anchor_slot_one_hot": tuple(oracle_anchor_slot_one_hot.shape),
        "state_reset": tuple(state_reset.shape),
        "model_evidence_delay_seconds": tuple(model_evidence_delay_seconds.shape),
    }
    if actual != expected:
        raise ValueError(f"PSEM feature shapes differ from the frozen interface: {actual}")
    _require_binary("slot alive indicators", slot_alive)
    _require_binary("oracle anchor slot indicators", oracle_anchor_slot_one_hot)
    _require_binary("state reset indicators", state_reset)
    one_hot = torch.all(oracle_anchor_slot_one_hot.sum(dim=-1) == 1)
    if one_hot.device.type == "cuda":
        torch._assert_async(one_hot, "oracle anchor slot indicators must be one-hot")
    elif not bool(one_hot):
        raise ValueError("oracle anchor slot indicators must be one-hot")
    if not torch.is_floating_point(model_evidence_delay_seconds):
        raise ValueError("model evidence delay must use a floating-point dtype")
    expected_delay = torch.full_like(model_evidence_delay_seconds, MODEL_EVIDENCE_DELAY_SECONDS)
    delay_matches = torch.all(
        torch.isclose(
            model_evidence_delay_seconds,
            expected_delay,
            rtol=0,
            atol=1e-6,
        )
    )
    if delay_matches.device.type == "cuda":
        torch._assert_async(delay_matches, "model evidence delay differs from the frozen interface")
    elif not bool(delay_matches):
        raise ValueError("model evidence delay differs from the frozen interface")
    dtype = final_temporal_hidden.dtype
    alive = slot_alive.to(dtype=dtype)
    anchor = oracle_anchor_slot_one_hot.to(dtype=dtype)
    selected_anchor_logit = (speaker_activity_logits * anchor).sum(dim=-1, keepdim=True)
    non_anchor_alive = torch.logical_and(slot_alive.bool(), torch.logical_not(anchor.bool()))
    negative_infinity = torch.full_like(speaker_activity_logits, -torch.inf)
    alive_non_anchor_logits = torch.where(
        non_anchor_alive, speaker_activity_logits, negative_infinity
    )
    maximum_alive_non_anchor_logit = alive_non_anchor_logits.max(dim=-1, keepdim=True).values
    maximum_alive_non_anchor_logit = torch.where(
        torch.isfinite(maximum_alive_non_anchor_logit),
        maximum_alive_non_anchor_logit,
        torch.zeros_like(maximum_alive_non_anchor_logit),
    )
    features = torch.cat(
        (
            final_temporal_hidden,
            speaker_activity_logits,
            alive,
            anchor,
            selected_anchor_logit,
            maximum_alive_non_anchor_logit,
            state_reset.to(dtype=dtype),
            model_evidence_delay_seconds.to(dtype=dtype),
        ),
        dim=-1,
    )
    if features.shape[-1] != PSEM_INPUT_DIMENSION:
        raise ValueError("PSEM feature dimension differs from the frozen interface")
    return features


class PSEMHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(PSEM_INPUT_DIMENSION)
        self.gru = nn.GRU(PSEM_INPUT_DIMENSION, 64, num_layers=1, batch_first=True)
        self.anchor_present = nn.Linear(64, 1)
        self.replacement_evidence = nn.Linear(64, 1)

    def forward(
        self,
        features: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        sequence_start_reset_only: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != PSEM_INPUT_DIMENSION:
            raise ValueError("PSEM head requires [batch, frames, 208] input")
        reset = features[..., -2]
        _require_binary("state reset indicators", reset)
        normalized = self.input_norm(features)
        if sequence_start_reset_only:
            start_only = torch.logical_and(
                reset[:, 0].bool().all(),
                torch.logical_not(reset[:, 1:].bool().any()),
            )
            if start_only.device.type == "cuda":
                torch._assert_async(
                    start_only,
                    "sequence reset indicators must occur only at the first frame",
                )
            elif not bool(start_only):
                raise ValueError("sequence reset indicators must occur only at the first frame")
            hidden, next_state = self.gru(normalized)
        elif not torch.any(reset):
            hidden, next_state = self.gru(normalized, state)
        else:
            frame_outputs = []
            next_state = state
            for frame_index in range(normalized.shape[1]):
                frame_reset = reset[:, frame_index].reshape(1, -1, 1)
                if next_state is not None:
                    next_state = next_state * (1 - frame_reset.to(dtype=next_state.dtype))
                frame_output, next_state = self.gru(
                    normalized[:, frame_index : frame_index + 1], next_state
                )
                frame_outputs.append(frame_output)
            hidden = torch.cat(frame_outputs, dim=1)
        return {
            "anchor_present": self.anchor_present(hidden).squeeze(-1),
            "replacement_evidence": self.replacement_evidence(hidden).squeeze(-1),
        }, next_state


def masked_balanced_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("logits, targets, and mask must have identical shapes")
    if positive_weight <= 0:
        raise ValueError("TRAIN-derived positive weight must be positive")
    _require_binary("targets", targets)
    _require_binary("loss mask", mask)
    losses = F.binary_cross_entropy_with_logits(
        logits,
        targets.to(dtype=logits.dtype),
        pos_weight=torch.as_tensor(positive_weight, dtype=logits.dtype, device=logits.device),
        reduction="none",
    )
    active = mask.to(dtype=logits.dtype)
    flattened_losses = (losses * active).flatten(start_dim=1)
    flattened_active = active.flatten(start_dim=1)
    numerator = flattened_losses.sum(dim=1)
    denominator = flattened_active.sum(dim=1).clamp_min(1)
    return (numerator / denominator).mean()


def composite_loss(
    outputs: dict[str, torch.Tensor],
    *,
    replacement_targets: torch.Tensor,
    anchor_targets: torch.Tensor,
    mask: torch.Tensor,
    replacement_positive_weight: float,
    anchor_positive_weight: float,
    sampling_roles: Sequence[str],
    native_sortformer_loss: NativeSortformerLoss,
) -> dict[str, Any]:
    roles = _loss_roles(sampling_roles)
    if mask.ndim < 1 or mask.shape[0] != len(roles):
        raise ValueError("sampling role count must match the loss batch")
    if (
        not isinstance(native_sortformer_loss, NativeSortformerLoss)
        or native_sortformer_loss._token is not _NATIVE_LOSS_TOKEN
        or native_sortformer_loss.sampling_roles != roles
    ):
        raise ValueError("native Sortformer loss lacks exact TRAIN provenance")
    replacement = masked_balanced_bce_with_logits(
        outputs["replacement_evidence"],
        replacement_targets,
        mask,
        replacement_positive_weight,
    )
    anchor = masked_balanced_bce_with_logits(
        outputs["anchor_present"], anchor_targets, mask, anchor_positive_weight
    )
    native_value = native_sortformer_loss.value
    total = replacement + 0.5 * anchor + 0.5 * native_value
    return {
        "total": total,
        "replacement": replacement,
        "anchor": anchor,
        "native_sortformer": native_value,
    }
