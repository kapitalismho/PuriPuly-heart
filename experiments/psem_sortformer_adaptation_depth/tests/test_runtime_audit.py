from __future__ import annotations

import copy
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.psem_sortformer_adaptation_depth.models import PSEMHead
from experiments.psem_sortformer_adaptation_depth.nemo_adapter import TrainableSortformerPSEM
from experiments.psem_sortformer_adaptation_depth.parameter_policy import (
    audit_parameter_graph,
    should_train,
)
from experiments.psem_sortformer_adaptation_depth.preflight import canonical_sha256
from experiments.psem_sortformer_adaptation_depth.runtime_audit import (
    LEARNING_RATES,
    build_timing_receipt,
    canary_bundle_runtime_passed,
    gradient_canary_runtime_passed,
    parameter_inventory,
    parameter_inventory_runtime_passed,
    run_gradient_update_canary,
    validate_streaming_graph,
)


class FakeFeatureLayer(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((dimension,), 0.9))
        self.bias = nn.Parameter(torch.full((dimension,), 0.1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.weight + self.bias


class FakeSortformerModules(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first_hidden_to_hidden = FakeFeatureLayer(192)
        self.single_hidden_to_spks = nn.Linear(192, 4)
        self.dropout = nn.Identity()
        self.chunk_len = 6
        self.chunk_right_context = 7
        self.fifo_len = 188
        self.spkcache_update_period = 144
        self.spkcache_len = 188
        self.chunk_left_context = 1
        self.n_spk = 4
        self.causal_attn_rate = 0.0
        self.causal_attn_rc = 7

    def length_to_mask(self, lengths: torch.Tensor, frame_count: int) -> torch.Tensor:
        positions = torch.arange(frame_count, device=lengths.device)
        return positions.unsqueeze(0) < lengths.unsqueeze(1)

    def init_streaming_state(
        self, *, batch_size: int, async_streaming: bool, device: torch.device
    ) -> SimpleNamespace:
        empty = torch.empty((batch_size, 0, 192), device=device)
        empty_lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
        return SimpleNamespace(
            spkcache=empty,
            fifo=empty,
            spkcache_lengths=empty_lengths,
            fifo_lengths=empty_lengths,
            spk_perm=None,
        )

    def streaming_feat_loader(
        self,
        *,
        feat_seq: torch.Tensor,
        feat_seq_length: torch.Tensor,
        feat_seq_offset: torch.Tensor,
    ):
        yield feat_seq_offset, feat_seq, feat_seq_length, 0, 0

    def concat_embs(
        self, values: list[torch.Tensor], *, dim: int, device: torch.device
    ) -> torch.Tensor:
        return torch.cat(values, dim=dim).to(device)

    def apply_mask_to_preds(self, predictions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return predictions * self.length_to_mask(lengths, predictions.shape[1]).unsqueeze(-1)

    def streaming_update(
        self,
        *,
        streaming_state: SimpleNamespace,
        chunk: torch.Tensor,
        preds: torch.Tensor,
        lc: int,
        rc: int,
    ) -> tuple[SimpleNamespace, torch.Tensor]:
        return streaming_state, preds


class FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeFeatureLayer(192) for _ in range(18)])

    def forward(
        self,
        value: torch.Tensor | None = None,
        *,
        encoder_states: torch.Tensor | None = None,
        encoder_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = encoder_states if encoder_states is not None else value
        if value is None:
            raise RuntimeError("transformer input is absent")
        for layer in self.layers:
            value = layer(value)
        return value


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.subsampling_factor = 1
        self.att_context_size = [-1, -1]

    def pre_encode(
        self, *, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, lengths


class FakeFrontendEncoder(nn.Module):
    def forward(
        self,
        *,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor,
        bypass_pre_encode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return processed_signal, processed_signal_length


class FakeSortformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_encoder = FakeTransformer()
        self.sortformer_modules = FakeSortformerModules()
        self.encoder = FakeEncoder()
        self.frontend_encoder = FakeFrontendEncoder()
        self.async_streaming = False
        self.device = torch.device("cpu")
        self._cfg = SimpleNamespace(model_defaults=SimpleNamespace(tf_d_model=192))

    def process_signal(
        self, *, audio_signal: torch.Tensor, audio_signal_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        basis = torch.linspace(0.5, 1.5, 192, device=audio_signal.device)
        frames = audio_signal[:, ::1280].unsqueeze(-1) * basis
        return frames, audio_signal_length // 1280


def _fake_adaptation_model() -> TrainableSortformerPSEM:
    with torch.random.fork_rng():
        torch.manual_seed(107)
        model = TrainableSortformerPSEM(FakeSortformer(), lambda *args, **kwargs: None)
    for parameter in model.sortformer.parameters():
        nn.init.constant_(parameter, 0.9)
    return model


class FakeGraphSortformerModules(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first_hidden_to_hidden = nn.Linear(192, 192)
        self.single_hidden_to_spks = nn.Linear(192, 4)
        self.chunk_len = 6
        self.chunk_right_context = 7
        self.fifo_len = 188
        self.spkcache_update_period = 144
        self.spkcache_len = 188
        self.chunk_left_context = 1
        self.n_spk = 4


class FakeGraphTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(192, 192) for _ in range(18)])


class FakeGraphSortformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_encoder = FakeGraphTransformer()
        self.sortformer_modules = FakeGraphSortformerModules()
        self._cfg = SimpleNamespace(model_defaults=SimpleNamespace(tf_d_model=192))


class FakeGraphAdaptationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sortformer = FakeGraphSortformer()
        self.psem_head = PSEMHead()
        self.runtime_taps = nn.ModuleDict(
            {
                "final_temporal_hidden": nn.Identity(),
                "speaker_activity_logits": nn.Identity(),
            }
        )


def test_model_graph_receipt_binds_low_latency_geometry_and_taps() -> None:
    model = FakeGraphAdaptationModel()
    receipt = validate_streaming_graph(model)
    assert receipt["passed"]
    assert receipt["streaming_geometry"]["chunk_len"] == 6
    assert receipt["hidden_tensor_identity"] == "sortformer.transformer_encoder.output"
    assert receipt["activity_logit_identity"].endswith("output_pre_sigmoid")
    assert len(receipt["executable_graph_sha256"]) == 64
    assert (
        receipt["parameter_schema_sha256"]
        == parameter_inventory(model, "F0-FROZEN-FLOAT")["parameter_schema_sha256"]
    )


def test_model_graph_rejects_config_claims_that_disagree_with_executable_shapes() -> None:
    model = FakeGraphAdaptationModel()
    model.sortformer._cfg.model_defaults.tf_d_model = 1
    with pytest.raises(Exception, match="hidden dimension"):
        validate_streaming_graph(model)


def _reseal_inventory(inventory: dict) -> dict:
    arm = inventory["arm"]
    rows = inventory["parameters"]
    names = [row["name"] for row in rows]
    trainable_counts = Counter()
    for row in rows:
        if row["requires_grad"]:
            trainable_counts[row["optimizer_group"]] += row["numel"]
    parameter_schema = [
        {"name": row["name"], "shape": row["shape"], "dtype": row["dtype"]} for row in rows
    ]
    payload = {
        "schema_version": 1,
        "artifact_role": "parameter_inventory",
        "arm": arm,
        "parameters": rows,
        "parameter_count": len(rows),
        "total_parameter_count": sum(row["numel"] for row in rows),
        "trainable_parameter_count": sum(row["numel"] for row in rows if row["requires_grad"]),
        "trainable_count_by_group": dict(sorted(trainable_counts.items())),
        "parameter_schema_sha256": canonical_sha256(parameter_schema),
        "policy": {
            **audit_parameter_graph(names),
            "arm": arm,
            "trainable": [name for name in names if should_train(name, arm)],
        },
    }
    return {**payload, "payload_sha256": canonical_sha256(payload)}


def test_parameter_inventory_is_recomputed_and_bound_to_the_executable_graph() -> None:
    model = FakeGraphAdaptationModel()
    graph = validate_streaming_graph(model)
    inventory = parameter_inventory(model, "T2-TOP")
    assert parameter_inventory_runtime_passed(
        inventory,
        "T2-TOP",
        model_graph_receipt=graph,
    )
    forged = copy.deepcopy(inventory)
    forged["parameters"] = [
        row
        for row in forged["parameters"]
        if row["name"] != "sortformer.transformer_encoder.layers.0.bias"
    ]
    forged = _reseal_inventory(forged)
    assert not parameter_inventory_runtime_passed(
        forged,
        "T2-TOP",
        model_graph_receipt=graph,
    )


def test_parameter_inventory_and_one_step_canaries_are_exact() -> None:
    model = _fake_adaptation_model()
    inventory = parameter_inventory(model, "T2-TOP")
    groups = {row["optimizer_group"] for row in inventory["parameters"] if row["requires_grad"]}
    assert groups == set(LEARNING_RATES)
    waveform = torch.linspace(-0.5, 0.5, 480000).unsqueeze(0)
    receipts = run_gradient_update_canary(model, "T2-TOP", waveform)
    assert receipts["gradient_canary_receipt"]["passed"]
    assert receipts["update_canary_receipt"]["passed"]
    assert receipts["update_canary_receipt"]["frozen_parameters_unchanged"]
    assert gradient_canary_runtime_passed(
        receipts["gradient_canary_receipt"],
        "T2-TOP",
        parameter_inventory_receipt=receipts["parameter_inventory"],
        model_graph_receipt=receipts["model_graph_receipt"],
    )
    timing = build_timing_receipt(
        torch.tensor([480000]),
        torch.full((1, 375, 4), 0.5),
        torch.zeros((1, 375, 4)),
        torch.zeros((1, 375, 192)),
        torch.ones((1, 375, 4)),
        torch.cat((torch.ones((1, 1, 1)), torch.zeros((1, 374, 1))), dim=1),
        torch.full((1, 375, 1), 1.04),
    )
    assert canary_bundle_runtime_passed(
        receipts["gradient_canary_receipt"],
        receipts["update_canary_receipt"],
        timing,
        "T2-TOP",
        parameter_inventory_receipt=receipts["parameter_inventory"],
        model_graph_receipt=receipts["model_graph_receipt"],
    )
    forged_update = {
        **receipts["update_canary_receipt"],
        "changed_parameters": [],
    }
    assert not canary_bundle_runtime_passed(
        receipts["gradient_canary_receipt"],
        forged_update,
        timing,
        "T2-TOP",
        parameter_inventory_receipt=receipts["parameter_inventory"],
        model_graph_receipt=receipts["model_graph_receipt"],
    )
    assert not gradient_canary_runtime_passed(
        {
            "raw_waveform_dependence": {},
            "module_reach_counts": {},
            "raw_waveform_gradient_nonzero": True,
        },
        "T2-TOP",
        parameter_inventory_receipt=receipts["parameter_inventory"],
        model_graph_receipt=receipts["model_graph_receipt"],
    )


def test_raw_waveform_canary_rejects_a_constant_graph_input() -> None:
    model = _fake_adaptation_model()

    def constant_runtime_canary_loss(waveform: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        value = torch.ones((waveform.shape[0], 375, 192), device=waveform.device)
        value = model.sortformer.transformer_encoder(value)
        value = model.runtime_taps["final_temporal_hidden"](value)
        value = model.sortformer.sortformer_modules.first_hidden_to_hidden(value)
        value = model.sortformer.sortformer_modules.single_hidden_to_spks(value)
        value = model.runtime_taps["speaker_activity_logits"](value)
        return model.psem_head(value).square().mean()

    model.runtime_canary_loss = constant_runtime_canary_loss

    with pytest.raises(Exception, match="exact fixed runtime canary path"):
        run_gradient_update_canary(model, "T2-TOP", torch.full((1, 480000), 0.5))


def test_runtime_canary_rejects_a_replaced_class_implementation(monkeypatch) -> None:
    model = _fake_adaptation_model()
    monkeypatch.setattr(
        TrainableSortformerPSEM,
        "runtime_canary_loss",
        lambda self, waveform, lengths: waveform.square().mean(),
    )
    with pytest.raises(Exception, match="exact fixed runtime canary path"):
        run_gradient_update_canary(model, "T2-TOP", torch.full((1, 480000), 0.5))


def test_timing_receipt_binds_exact_frame_count_delay_and_binary_lifecycle() -> None:
    lengths = torch.tensor([2560])
    probabilities = torch.full((1, 2, 4), 0.5)
    logits = torch.zeros((1, 2, 4))
    hidden = torch.zeros((1, 2, 192))
    alive = torch.ones((1, 2, 4))
    reset = torch.tensor([[[1.0], [0.0]]])
    delay = torch.full((1, 2, 1), 1.04)
    receipt = build_timing_receipt(lengths, probabilities, logits, hidden, alive, reset, delay)
    assert receipt["frame_counts"] == [2]
    with pytest.raises(Exception, match="native frame contract"):
        build_timing_receipt(
            lengths,
            probabilities,
            logits,
            hidden,
            alive,
            torch.zeros_like(reset),
            delay,
        )
    with pytest.raises(Exception, match="native frame contract"):
        build_timing_receipt(
            lengths,
            probabilities[:, :1],
            logits[:, :1],
            hidden[:, :1],
            alive[:, :1],
            reset[:, :1],
            delay[:, :1],
        )
