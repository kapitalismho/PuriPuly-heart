from __future__ import annotations

import math
from collections.abc import Sequence


CLAMP_LO = 1e-6
CLAMP_HI = 1.0 - 1e-6
IDENTITY_TOL = 1e-6
GRU_HIDDEN = 64


def clamp_prob(value: float) -> float:
    return min(max(value, CLAMP_LO), CLAMP_HI)


def logit(prob: float) -> float:
    clamped = clamp_prob(prob)
    return math.log(clamped / (1.0 - clamped))


def sigmoid(value: float) -> float:
    clipped = min(max(value, -80.0), 80.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def f0_logit(selected_anchor_posterior: float) -> float:
    return logit(1.0 - clamp_prob(selected_anchor_posterior))


def product_logit(selected_anchor_posterior: float, z_residual: float) -> float:
    return f0_logit(selected_anchor_posterior) + z_residual


def check_zero_residual_identity(
    posteriors: Sequence[float], tol: float = IDENTITY_TOL
) -> dict[str, object]:
    diffs = [abs(product_logit(p, 0.0) - f0_logit(p)) for p in posteriors]
    worst = max(diffs, default=0.0)
    return {"max_abs_diff": worst, "tol": tol, "passed": worst <= tol}


def selective_mode_receipt(backbone_eval: bool, head_train: bool) -> dict[str, object]:
    receipt = {"sortformer_eval": backbone_eval, "psem_head_train": head_train}
    receipt["frozen_representation_ok"] = bool(backbone_eval and head_train)
    return receipt


def audit_update(
    frozen_before: Sequence[Sequence[float]],
    frozen_after: Sequence[Sequence[float]],
    head_grads: Sequence[Sequence[float] | None],
    head_before: Sequence[Sequence[float]],
    head_after: Sequence[Sequence[float]],
) -> dict[str, object]:
    frozen_moved = any(
        a != b for row_a, row_b in zip(frozen_before, frozen_after) for a, b in zip(row_a, row_b)
    )
    finite_grads = [
        all(math.isfinite(v) for v in g) for g in head_grads if g is not None
    ]
    head_moved = any(
        a != b for row_a, row_b in zip(head_before, head_after) for a, b in zip(row_a, row_b)
    )
    passed = (not frozen_moved) and len(finite_grads) > 0 and all(finite_grads) and head_moved
    return {
        "frozen_unchanged": not frozen_moved,
        "trainable_grads_finite": len(finite_grads) > 0 and all(finite_grads),
        "trainable_changed": head_moved,
        "passed": passed,
    }


try:
    import torch
    from torch import nn

    class ResidualPSEMHead(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(input_dim)
            self.gru = nn.GRU(input_dim, GRU_HIDDEN, num_layers=1, batch_first=True)
            self.anchor_out = nn.Linear(GRU_HIDDEN, 1)
            self.residual_out = nn.Linear(GRU_HIDDEN, 1)
            nn.init.zeros_(self.residual_out.weight)
            nn.init.zeros_(self.residual_out.bias)

        def forward(self, features, state=None):
            normed = self.norm(features)
            outputs, next_state = self.gru(normed, state)
            return {
                "anchor_logit": self.anchor_out(outputs).squeeze(-1),
                "z_residual": self.residual_out(outputs).squeeze(-1),
            }, next_state
except ImportError:
    ResidualPSEMHead = None
