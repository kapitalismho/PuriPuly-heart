from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class StreamingState:
    sample_buffer: Tensor
    conv_buffers: tuple[Tensor, ...]
    hidden: Tensor
    total_samples: int
    emitted_outputs: int

    def detached(self) -> "StreamingState":
        return StreamingState(
            sample_buffer=self.sample_buffer.detach(),
            conv_buffers=tuple(value.detach() for value in self.conv_buffers),
            hidden=self.hidden.detach(),
            total_samples=self.total_samples,
            emitted_outputs=self.emitted_outputs,
        )

    def cpu_dict(self) -> dict[str, object]:
        return {
            "sample_buffer": self.sample_buffer.detach().cpu(),
            "conv_buffers": [value.detach().cpu() for value in self.conv_buffers],
            "hidden": self.hidden.detach().cpu(),
            "total_samples": self.total_samples,
            "emitted_outputs": self.emitted_outputs,
        }


class CausalConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, stride=2)
        self.activation = nn.SiLU()

    def forward_stream(self, values: Tensor, pending: Tensor) -> tuple[Tensor, Tensor]:
        combined = torch.cat((pending, values), dim=2)
        if combined.shape[2] < self.kernel_size:
            return values.new_empty((values.shape[0], self.conv.out_channels, 0)), combined
        output_count = 1 + (combined.shape[2] - self.kernel_size) // 2
        consumed = output_count * 2
        output = self.activation(self.conv(combined))
        return output, combined[:, :, consumed:]


class StreamingStudent(nn.Module):
    sample_rate = 16000
    window_samples = 400
    mel_hop_samples = 160
    output_hop_samples = 1280
    conv_kernel_size = 5
    conv_strides = (2, 2, 2)
    mel_bins = 64
    conv_channels = 512
    hidden_size = 512
    gru_layers = 2
    output_slots = 4

    def __init__(self) -> None:
        super().__init__()
        channels = (self.mel_bins, self.conv_channels, self.conv_channels, self.conv_channels)
        self.conv_blocks = nn.ModuleList(
            CausalConvBlock(channels[index], channels[index + 1], self.conv_kernel_size)
            for index in range(3)
        )
        self.gru = nn.GRU(self.conv_channels, self.hidden_size, self.gru_layers, batch_first=True)
        self.output = nn.Linear(self.hidden_size, self.output_slots)
        self.register_buffer(
            "analysis_window", torch.hann_window(self.window_samples), persistent=True
        )
        self.register_buffer("mel_filter", self._mel_filter(), persistent=True)

    @staticmethod
    def _hz_to_mel(value: float) -> float:
        return 2595.0 * math.log10(1.0 + value / 700.0)

    @staticmethod
    def _mel_to_hz(value: float) -> float:
        return 700.0 * (10.0 ** (value / 2595.0) - 1.0)

    def _mel_filter(self) -> Tensor:
        frequency_bins = self.window_samples // 2 + 1
        low = self._hz_to_mel(20.0)
        high = self._hz_to_mel(self.sample_rate / 2)
        edges = torch.linspace(low, high, self.mel_bins + 2)
        hz_edges = torch.tensor([self._mel_to_hz(float(value)) for value in edges])
        frequencies = torch.linspace(0.0, self.sample_rate / 2, frequency_bins)
        filters = torch.zeros(self.mel_bins, frequency_bins)
        for index in range(self.mel_bins):
            left, center, right = hz_edges[index : index + 3]
            filters[index] = torch.minimum(
                (frequencies - left) / (center - left),
                (right - frequencies) / (right - center),
            ).clamp_min(0.0)
        return filters

    @property
    def receptive_field_samples(self) -> int:
        mel_frames = 1 + (self.conv_kernel_size - 1) * sum(
            math.prod(self.conv_strides[:index]) for index in range(len(self.conv_strides))
        )
        return self.window_samples + (mel_frames - 1) * self.mel_hop_samples

    def initial_state(self, batch_size: int, device: torch.device) -> StreamingState:
        dtype = next(self.parameters()).dtype
        buffers: list[Tensor] = []
        for block in self.conv_blocks:
            buffers.append(
                torch.zeros(
                    batch_size,
                    block.conv.in_channels,
                    block.kernel_size - block.conv.stride[0],
                    device=device,
                    dtype=dtype,
                )
            )
        return StreamingState(
            sample_buffer=torch.zeros(
                batch_size,
                self.window_samples - self.mel_hop_samples,
                device=device,
                dtype=dtype,
            ),
            conv_buffers=tuple(buffers),
            hidden=torch.zeros(
                self.gru_layers, batch_size, self.hidden_size, device=device, dtype=dtype
            ),
            total_samples=0,
            emitted_outputs=0,
        )

    def state_from_dict(self, value: dict[str, object], device: torch.device) -> StreamingState:
        return StreamingState(
            sample_buffer=value["sample_buffer"].to(device),
            conv_buffers=tuple(item.to(device) for item in value["conv_buffers"]),
            hidden=value["hidden"].to(device),
            total_samples=int(value["total_samples"]),
            emitted_outputs=int(value["emitted_outputs"]),
        )

    def _features(self, audio: Tensor, state: StreamingState) -> tuple[Tensor, Tensor]:
        combined = torch.cat((state.sample_buffer, audio), dim=1)
        if combined.shape[1] < self.window_samples:
            return audio.new_empty((audio.shape[0], self.mel_bins, 0)), combined
        frame_count = 1 + (combined.shape[1] - self.window_samples) // self.mel_hop_samples
        frames = combined.unfold(1, self.window_samples, self.mel_hop_samples)[:, :frame_count]
        spectrum = torch.fft.rfft(frames * self.analysis_window, dim=2)
        power = spectrum.real.square() + spectrum.imag.square()
        mel = torch.matmul(power, self.mel_filter.T)
        features = ((mel.clamp_min(1.0e-8).log() + 10.0) / 5.0).transpose(1, 2)
        consumed = frame_count * self.mel_hop_samples
        return features, combined[:, consumed:]

    def forward_stream(
        self, audio: Tensor, state: StreamingState
    ) -> tuple[Tensor, Tensor, StreamingState]:
        if audio.ndim != 2:
            raise ValueError("audio must have shape [batch, samples]")
        features, sample_buffer = self._features(audio, state)
        values = features
        next_buffers: list[Tensor] = []
        for block, pending in zip(self.conv_blocks, state.conv_buffers, strict=True):
            values, next_pending = block.forward_stream(values, pending)
            next_buffers.append(next_pending)
        if values.shape[2]:
            sequence = values.transpose(1, 2)
            encoded, hidden = self.gru(sequence, state.hidden)
            logits = self.output(encoded)
        else:
            logits = audio.new_empty((audio.shape[0], 0, self.output_slots))
            hidden = state.hidden
        first_index = state.emitted_outputs
        frontiers = (
            torch.arange(
                first_index,
                first_index + logits.shape[1],
                device=audio.device,
                dtype=torch.int64,
            )
            + 1
        ) * self.output_hop_samples
        next_state = StreamingState(
            sample_buffer=sample_buffer,
            conv_buffers=tuple(next_buffers),
            hidden=hidden,
            total_samples=state.total_samples + audio.shape[1],
            emitted_outputs=state.emitted_outputs + logits.shape[1],
        )
        return logits, frontiers, next_state
