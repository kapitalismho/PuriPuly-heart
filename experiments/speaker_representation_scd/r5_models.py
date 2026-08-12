from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LinearProbe(nn.Module):
    def __init__(self, input_dimension: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(input_dimension)
        self.output = nn.Linear(input_dimension, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(self.normalization(values)).squeeze(-1)


class CausalResidualBlock(nn.Module):
    def __init__(self, dimension: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            dimension,
            dimension,
            kernel_size,
            dilation=dilation,
        )
        self.normalization = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        convolved = self.convolution(F.pad(values.transpose(1, 2), (self.left_padding, 0)))
        activated = self.dropout(F.gelu(convolved.transpose(1, 2)))
        return self.normalization(values + activated)


class CausalTCN(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_normalization = nn.LayerNorm(input_dimension)
        self.projection = nn.Linear(input_dimension, hidden_dimension)
        self.blocks = nn.ModuleList(
            [
                CausalResidualBlock(hidden_dimension, kernel_size, dilation, dropout)
                for dilation in dilations
            ]
        )
        self.output = nn.Linear(hidden_dimension, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(self.input_normalization(values))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden).squeeze(-1)
