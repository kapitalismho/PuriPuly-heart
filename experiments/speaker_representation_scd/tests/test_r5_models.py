from __future__ import annotations

import torch

from experiments.speaker_representation_scd.r5_models import CausalTCN


def test_causal_tcn_does_not_use_future_frames() -> None:
    torch.manual_seed(0)
    model = CausalTCN(4, 8, 3, [1, 2, 4], 0.0).eval()
    first = torch.randn(1, 20, 4)
    second = first.clone()
    second[:, 12:] = torch.randn_like(second[:, 12:])
    with torch.no_grad():
        first_logits = model(first)
        second_logits = model(second)
    torch.testing.assert_close(first_logits[:, :12], second_logits[:, :12])


def test_causal_tcn_preserves_time_axis() -> None:
    model = CausalTCN(4, 8, 3, [1, 2, 4], 0.1)
    assert model(torch.randn(3, 17, 4)).shape == (3, 17)
