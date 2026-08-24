from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torchaudio

from experiments.psem_training_strategy_gate.targets import (
    CELL_COUNT,
    HOP_SAMPLES,
    SAMPLE_RATE_HZ,
    WINDOW_SAMPLES,
)

ARMS = ("FROZEN-WAVLM", "FINETUNE-WAVLM", "SCRATCH-PSEM")
WAVLM_MODEL_ID = "wavlm-base-plus"
WAVLM_REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"
WAVLM_FRAME_COUNT = 149
WAVLM_FRAME_STRIDE_SAMPLES = 320
WAVLM_RECEPTIVE_FIELD_SAMPLES = 400
SCRATCH_WIDTH = 320
SCRATCH_EXPANSION = 2
SCRATCH_DILATIONS = (1, 2, 4, 8, 16, 1, 2, 4)
CELL_DIMENSION = 256
COMMON_HEAD_LR = 1e-3
FINETUNED_WAVLM_LR = 1e-5
SCRATCH_ENCODER_LR = 3e-4


class ModelContractError(RuntimeError):
    pass


def _cell_indices(frame_count: int, stride_samples: int, receptive_field_samples: int):
    centers = (
        torch.arange(frame_count, dtype=torch.long) * stride_samples + receptive_field_samples // 2
    )
    indices = torch.div(centers, HOP_SAMPLES, rounding_mode="floor")
    if int(indices.min()) != 0 or int(indices.max()) != CELL_COUNT - 1:
        raise ModelContractError("encoder frames do not cover the exact 30-cell source grid")
    return indices


def _pool_source_aligned(
    hidden: torch.Tensor,
    *,
    stride_samples: int,
    receptive_field_samples: int,
) -> torch.Tensor:
    if hidden.ndim != 3:
        raise ModelContractError("encoder hidden states must be batch by frame by dimension")
    indices = _cell_indices(hidden.shape[1], stride_samples, receptive_field_samples).to(
        hidden.device
    )
    cells = [hidden[:, indices == cell].mean(dim=1) for cell in range(CELL_COUNT)]
    if any(torch.isnan(cell).any() for cell in cells):
        raise ModelContractError("one or more source-aligned cells have no encoder frames")
    return torch.stack(cells, dim=1)


class WavLMCellEncoder(torch.nn.Module):
    output_dimension = 768

    def __init__(self, model_root: Path, *, fine_tune: bool) -> None:
        super().__init__()
        from transformers import WavLMModel

        self.model_root = model_root.resolve()
        self.wavlm = WavLMModel.from_pretrained(
            self.model_root,
            local_files_only=True,
            use_safetensors=False,
        )
        config = self.wavlm.config
        observed = {
            "model_type": config.model_type,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "conv_kernel": list(config.conv_kernel),
            "conv_stride": list(config.conv_stride),
            "do_stable_layer_norm": config.do_stable_layer_norm,
            "output_frames": int(
                self.wavlm._get_feat_extract_output_lengths(torch.tensor([WINDOW_SAMPLES]))[0]
            ),
        }
        expected = {
            "model_type": "wavlm",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
            "conv_stride": [5, 2, 2, 2, 2, 2, 2],
            "do_stable_layer_norm": False,
            "output_frames": WAVLM_FRAME_COUNT,
        }
        if observed != expected:
            raise ModelContractError(f"pinned WavLM graph differs: {observed}")
        for parameter in self.wavlm.parameters():
            parameter.requires_grad = False
        if fine_tune:
            for layer in self.wavlm.encoder.layers[8:12]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            for parameter in self.wavlm.encoder.layer_norm.parameters():
                parameter.requires_grad = True
        else:
            self.wavlm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.wavlm.parameters()):
            self.wavlm.eval()
        return self

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mean = waveform.mean(dim=1, keepdim=True)
        variance = waveform.var(dim=1, keepdim=True, unbiased=False)
        normalized = (waveform - mean) / torch.sqrt(variance + 1e-7)
        if any(parameter.requires_grad for parameter in self.wavlm.parameters()):
            hidden = self.wavlm(normalized).last_hidden_state
        else:
            with torch.no_grad():
                hidden = self.wavlm(normalized).last_hidden_state
        if hidden.shape[1:] != (WAVLM_FRAME_COUNT, self.output_dimension):
            raise ModelContractError("WavLM emitted an unexpected hidden-state geometry")
        return _pool_source_aligned(
            hidden,
            stride_samples=WAVLM_FRAME_STRIDE_SAMPLES,
            receptive_field_samples=WAVLM_RECEPTIVE_FIELD_SAMPLES,
        )


class GatedTemporalBlock(torch.nn.Module):
    def __init__(self, width: int, expansion: int, kernel: int, dilation: int) -> None:
        super().__init__()
        expanded = width * expansion
        self.normalization = torch.nn.GroupNorm(1, width)
        self.expand = torch.nn.Conv1d(width, expanded * 2, 1)
        self.depthwise = torch.nn.Conv1d(
            expanded,
            expanded,
            kernel,
            padding=dilation * (kernel - 1) // 2,
            dilation=dilation,
            groups=expanded,
        )
        self.project = torch.nn.Conv1d(expanded, width, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.normalization(values)
        content, gate = self.expand(values).chunk(2, dim=1)
        values = torch.nn.functional.silu(content) * torch.sigmoid(gate)
        values = torch.nn.functional.silu(self.depthwise(values))
        return residual + self.project(values)


class ScratchCellEncoder(torch.nn.Module):
    output_dimension = SCRATCH_WIDTH

    def __init__(self) -> None:
        super().__init__()
        self.frontend = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE_HZ,
            n_fft=400,
            win_length=400,
            hop_length=160,
            center=False,
            power=2.0,
            n_mels=64,
            norm="slaney",
            mel_scale="slaney",
        )
        self.stem = torch.nn.Conv1d(64, SCRATCH_WIDTH, 5, padding=2)
        self.blocks = torch.nn.ModuleList(
            GatedTemporalBlock(SCRATCH_WIDTH, SCRATCH_EXPANSION, 5, dilation)
            for dilation in SCRATCH_DILATIONS
        )
        self.final_normalization = torch.nn.GroupNorm(1, SCRATCH_WIDTH)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrum = self.frontend(waveform).clamp_min(1e-10).log()
        values = torch.nn.functional.silu(self.stem(spectrum))
        for block in self.blocks:
            values = block(values)
        hidden = self.final_normalization(values).transpose(1, 2)
        return _pool_source_aligned(
            hidden,
            stride_samples=160,
            receptive_field_samples=400,
        )


class CommonPSEMHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temporal = torch.nn.GRU(
            CELL_DIMENSION,
            128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.handoff_head = torch.nn.Sequential(
            torch.nn.Linear(CELL_DIMENSION * 4, CELL_DIMENSION),
            torch.nn.GELU(),
            torch.nn.Linear(CELL_DIMENSION, 1),
        )
        self.state_head = torch.nn.Linear(CELL_DIMENSION, 3)
        self.relation_head = torch.nn.Sequential(
            torch.nn.Linear(CELL_DIMENSION * 2, CELL_DIMENSION),
            torch.nn.GELU(),
            torch.nn.Linear(CELL_DIMENSION, 1),
        )

    def forward(self, cells: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden, _ = self.temporal(cells)
        left = hidden[:, 19]
        right = hidden[:, 20]
        boundary_features = torch.cat(
            (left, right, torch.abs(left - right), left * right),
            dim=-1,
        )
        return {
            "hidden": hidden,
            "handoff_logits": self.handoff_head(boundary_features).squeeze(-1),
            "state_logits": self.state_head(hidden),
        }

    def relation_logits(
        self,
        hidden: torch.Tensor,
        batch_indices: torch.Tensor,
        left_cells: torch.Tensor,
        right_cells: torch.Tensor,
    ) -> torch.Tensor:
        left = hidden[batch_indices, left_cells]
        right = hidden[batch_indices, right_cells]
        features = torch.cat((torch.abs(left - right), left * right), dim=-1)
        return self.relation_head(features).squeeze(-1)


class PSEMModel(torch.nn.Module):
    def __init__(self, arm: str, encoder: torch.nn.Module, projection_input: int) -> None:
        super().__init__()
        self.arm = arm
        self.encoder = encoder
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(projection_input, CELL_DIMENSION),
            torch.nn.GELU(),
            torch.nn.LayerNorm(CELL_DIMENSION),
        )
        self.head = CommonPSEMHead()

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        if waveform.ndim != 2 or waveform.shape[1] != WINDOW_SAMPLES:
            raise ModelContractError("model input must be raw 16 kHz three-second waveforms")
        cells = self.projection(self.encoder(waveform))
        if cells.shape[1:] != (CELL_COUNT, CELL_DIMENSION):
            raise ModelContractError("encoder projection differs from the common cell contract")
        outputs = self.head(cells)
        outputs["cells"] = cells
        return outputs


def model_root(cache_root: Path) -> Path:
    return cache_root.resolve() / "models" / WAVLM_MODEL_ID / WAVLM_REVISION


def build_model(arm: str, *, cache_root: Path, seed: int) -> PSEMModel:
    if arm not in ARMS:
        raise ModelContractError(f"unknown official arm: {arm}")
    if arm == "SCRATCH-PSEM":
        with torch.random.fork_rng():
            torch.manual_seed(seed + 1000)
            encoder: torch.nn.Module = ScratchCellEncoder()
        projection_input = SCRATCH_WIDTH
    else:
        encoder = WavLMCellEncoder(
            model_root(cache_root),
            fine_tune=arm == "FINETUNE-WAVLM",
        )
        projection_input = WavLMCellEncoder.output_dimension
    with torch.random.fork_rng():
        torch.manual_seed(seed + 2000)
        projection = torch.nn.Sequential(
            torch.nn.Linear(projection_input, CELL_DIMENSION),
            torch.nn.GELU(),
            torch.nn.LayerNorm(CELL_DIMENSION),
        )
    with torch.random.fork_rng():
        torch.manual_seed(seed + 3000)
        head = CommonPSEMHead()
    model = PSEMModel(arm, encoder, projection_input)
    model.projection = projection
    model.head = head
    return model


def wavlm_parameter_allowed(name: str) -> bool:
    return name.startswith(
        tuple(f"encoder.wavlm.encoder.layers.{index}." for index in range(8, 12))
    ) or name.startswith("encoder.wavlm.encoder.layer_norm.")


@dataclass(frozen=True, slots=True)
class OptimizerGroup:
    name: str
    learning_rate: float
    parameter_names: tuple[str, ...]


def optimizer_groups(model: PSEMModel) -> tuple[OptimizerGroup, ...]:
    names = {name: parameter for name, parameter in model.named_parameters()}
    common = tuple(
        name
        for name, parameter in names.items()
        if parameter.requires_grad and (name.startswith("projection.") or name.startswith("head."))
    )
    groups = [OptimizerGroup("common_head_and_projection", COMMON_HEAD_LR, common)]
    if model.arm == "FINETUNE-WAVLM":
        groups.append(
            OptimizerGroup(
                "finetuned_wavlm",
                FINETUNED_WAVLM_LR,
                tuple(
                    name
                    for name, parameter in names.items()
                    if parameter.requires_grad and name.startswith("encoder.wavlm.")
                ),
            )
        )
    elif model.arm == "SCRATCH-PSEM":
        groups.append(
            OptimizerGroup(
                "scratch_encoder",
                SCRATCH_ENCODER_LR,
                tuple(
                    name
                    for name, parameter in names.items()
                    if parameter.requires_grad and name.startswith("encoder.")
                ),
            )
        )
    assigned = [name for group in groups for name in group.parameter_names]
    trainable = [name for name, parameter in names.items() if parameter.requires_grad]
    if len(assigned) != len(set(assigned)) or sorted(assigned) != sorted(trainable):
        raise ModelContractError("optimizer groups do not cover trainable parameters exactly once")
    return tuple(groups)


def build_optimizer(model: PSEMModel) -> torch.optim.Optimizer:
    parameters = dict(model.named_parameters())
    groups = [
        {
            "params": [parameters[name] for name in group.parameter_names],
            "lr": group.learning_rate,
            "group_name": group.name,
        }
        for group in optimizer_groups(model)
    ]
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def parameter_inventory(model: PSEMModel) -> dict[str, Any]:
    group_by_name = {
        name: group for group in optimizer_groups(model) for name in group.parameter_names
    }
    rows = []
    for name, parameter in model.named_parameters():
        group = group_by_name.get(name)
        rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "owner_module": name.rsplit(".", 1)[0],
                "trainable": parameter.requires_grad,
                "optimizer_group": group.name if group is not None else None,
                "learning_rate": group.learning_rate if group is not None else None,
            }
        )
    return {
        "arm": model.arm,
        "parameters": rows,
        "total_parameters": sum(row["numel"] for row in rows),
        "trainable_parameters": sum(row["numel"] for row in rows if row["trainable"]),
        "trainable_wavlm_parameters": sum(
            row["numel"]
            for row in rows
            if row["trainable"] and row["name"].startswith("encoder.wavlm.")
        ),
        "optimizer_groups": [
            {
                "name": group.name,
                "learning_rate": group.learning_rate,
                "parameter_count": sum(
                    parameters.numel()
                    for name, parameters in model.named_parameters()
                    if name in group.parameter_names
                ),
            }
            for group in optimizer_groups(model)
        ],
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def finite_nonzero_gradient(parameters: Iterable[torch.nn.Parameter]) -> bool:
    gradients = [parameter.grad for parameter in parameters]
    return bool(gradients) and all(
        gradient is not None
        and bool(torch.isfinite(gradient).all())
        and float(gradient.norm()) > 0.0
        for gradient in gradients
    )
