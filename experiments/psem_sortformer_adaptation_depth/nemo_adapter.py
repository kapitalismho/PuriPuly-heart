from __future__ import annotations

import copy
import importlib.metadata
import inspect
import json
import math
import platform
import random
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.psem_sortformer_adaptation_depth.models import (
    MODEL_EVIDENCE_DELAY_SECONDS,
    NATIVE_SORTFORMER_CHECKPOINT_SHA256,
    NATIVE_SORTFORMER_LOSS_KIND,
    NATIVE_SORTFORMER_LOSS_ORIGIN,
    PSEMHead,
    bind_native_sortformer_loss,
    build_psem_features,
)
from experiments.psem_sortformer_adaptation_depth.preflight import sha256_file
from experiments.psem_sortformer_adaptation_depth.runtime_audit import (
    LOW_LATENCY_STREAMING,
    validate_streaming_graph,
)

NEMO_REVISION = "1a3c291b3ef0f0e11b72f789b185e1f1bda39bd6"
FRAME_SAMPLES = 1280
REQUIRED_LOCK_PACKAGES = {
    "hydra-core",
    "lightning",
    "numpy",
    "omegaconf",
    "torch",
    "torchaudio",
}


class NeMoAdapterError(RuntimeError):
    pass


def _installed_dependency_inventory() -> list[dict[str, str]]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        version = distribution.version
        if not isinstance(raw_name, str) or not isinstance(version, str) or not version:
            raise NeMoAdapterError("installed distribution metadata is incomplete")
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        if name in packages and packages[name] != version:
            raise NeMoAdapterError(f"installed distribution name is ambiguous: {name}")
        packages[name] = version
    return [{"name": name, "version": packages[name]} for name in sorted(packages)]


def _platform_identity() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def build_dependency_lock() -> dict[str, Any]:
    packages = _installed_dependency_inventory()
    if not REQUIRED_LOCK_PACKAGES <= {row["name"] for row in packages}:
        raise NeMoAdapterError("the NeMo runtime dependency inventory is incomplete")
    return {
        "schema_version": 1,
        "artifact_role": "nemo_dependency_lock",
        "nemo_revision": NEMO_REVISION,
        "python_version": platform.python_version(),
        "platform": _platform_identity(),
        "lock_kind": "complete_installed_distribution_inventory",
        "packages": packages,
    }


def write_dependency_lock(path: Path) -> dict[str, Any]:
    value = build_dependency_lock()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**value, "path": str(path.resolve()), "sha256": sha256_file(path)}


def validate_dependency_lock(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise NeMoAdapterError("dependency lock is absent")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NeMoAdapterError("dependency lock is not canonical JSON") from exc
    packages = value.get("packages") if isinstance(value, dict) else None
    if (
        value.get("schema_version") != 1
        or value.get("artifact_role") != "nemo_dependency_lock"
        or value.get("nemo_revision") != NEMO_REVISION
        or value.get("python_version") != platform.python_version()
        or value.get("platform") != _platform_identity()
        or value.get("lock_kind") != "complete_installed_distribution_inventory"
        or not isinstance(packages, list)
    ):
        raise NeMoAdapterError("dependency lock identity is invalid")
    names: list[str] = []
    for row in packages:
        name = row.get("name") if isinstance(row, dict) else None
        version = row.get("version") if isinstance(row, dict) else None
        if (
            not isinstance(name, str)
            or name != name.lower()
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name) is None
            or not isinstance(version, str)
            or not version
        ):
            raise NeMoAdapterError("dependency lock contains an invalid exact package pin")
        names.append(name)
    if names != sorted(set(names)) or not REQUIRED_LOCK_PACKAGES <= set(names):
        raise NeMoAdapterError("dependency lock is incomplete or not canonically ordered")
    observed = _installed_dependency_inventory()
    if packages != observed:
        raise NeMoAdapterError("installed dependency inventory differs from lock")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "python_version": platform.python_version(),
        "packages": observed,
    }


def _assert_symbol_origin(symbol: Any, checkout: Path, name: str) -> str:
    implementation = Path(inspect.getfile(symbol)).resolve()
    if not implementation.is_relative_to(checkout):
        raise NeMoAdapterError(f"imported NeMo symbol is outside the frozen checkout: {name}")
    return str(implementation)


def _assert_loaded_nemo_origins(checkout: Path) -> list[dict[str, str]]:
    origins = []
    for name, module in sorted(sys.modules.items()):
        if name != "nemo" and not name.startswith("nemo."):
            continue
        raw = getattr(module, "__file__", None)
        if raw is None:
            continue
        path = Path(raw).resolve()
        if not path.is_relative_to(checkout):
            raise NeMoAdapterError(f"loaded NeMo module is outside the frozen checkout: {name}")
        origins.append({"module": name, "path": str(path)})
    if not origins:
        raise NeMoAdapterError("no frozen NeMo module origins were observed")
    return origins


def _validate_state_reset_lifecycle(
    state_reset: torch.Tensor, *, batch_size: int, frame_count: int
) -> None:
    if (
        state_reset.dtype != torch.bool
        or state_reset.shape != (batch_size, frame_count, 1)
        or not bool(state_reset[:, 0, 0].all())
        or (frame_count > 1 and bool(state_reset[:, 1:, 0].any()))
    ):
        raise NeMoAdapterError("state-reset evidence differs from actual sequence initialization")


@contextmanager
def _temporary_causal_attention(model: nn.Module, enabled: bool):
    original_attention_context = copy.deepcopy(model.encoder.att_context_size)
    diag_existed = hasattr(model.transformer_encoder, "diag")
    original_transformer_diag = copy.deepcopy(getattr(model.transformer_encoder, "diag", None))
    if enabled:
        model.encoder.att_context_size = [-1, model.sortformer_modules.causal_attn_rc]
        model.transformer_encoder.diag = model.sortformer_modules.causal_attn_rc
    try:
        yield
    finally:
        if enabled:
            model.encoder.att_context_size = original_attention_context
            if diag_existed:
                model.transformer_encoder.diag = original_transformer_diag
            else:
                delattr(model.transformer_encoder, "diag")


@dataclass(frozen=True, slots=True)
class SortformerEvidence:
    probabilities: torch.Tensor
    activity_logits: torch.Tensor
    final_temporal_hidden: torch.Tensor
    slot_alive: torch.Tensor
    state_reset: torch.Tensor
    evidence_delay_seconds: torch.Tensor


def compact_valid_frames(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        probabilities.shape != targets.shape
        or probabilities.ndim != 3
        or probabilities.shape[-1] != 4
        or target_mask.shape != probabilities.shape[:2]
        or target_mask.dtype != torch.bool
    ):
        raise NeMoAdapterError("native target and mask geometry differs from predictions")
    lengths = target_mask.sum(dim=1)
    if bool((lengths <= 0).any()):
        raise NeMoAdapterError("native loss requires valid frames in every sequence")
    maximum = int(lengths.max())
    compact_probabilities = probabilities.new_zeros((probabilities.shape[0], maximum, 4))
    compact_targets = targets.new_zeros((targets.shape[0], maximum, 4))
    for batch_index in range(targets.shape[0]):
        count = int(lengths[batch_index])
        compact_probabilities[batch_index, :count] = probabilities[
            batch_index, target_mask[batch_index]
        ]
        compact_targets[batch_index, :count] = targets[batch_index, target_mask[batch_index]]
    return compact_probabilities, compact_targets, lengths


def load_pinned_sortformer(
    checkpoint_path: Path,
    nemo_checkout: Path,
    dependency_lock: Path,
    device: torch.device | str,
) -> tuple[TrainableSortformerPSEM, dict[str, Any]]:
    if (
        not checkpoint_path.is_file()
        or sha256_file(checkpoint_path) != NATIVE_SORTFORMER_CHECKPOINT_SHA256
    ):
        raise NeMoAdapterError("trainable checkpoint identity differs from the frozen artifact")
    checkout = nemo_checkout.resolve()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if head != NEMO_REVISION or dirty:
        raise NeMoAdapterError("NeMo checkout is not the clean frozen revision")
    lock = validate_dependency_lock(dependency_lock)
    try:
        from nemo.collections.asr.models.sortformer_diar_models import (
            SortformerEncLabelModel,
        )
        from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_ats_targets
    except ImportError as exc:
        raise NeMoAdapterError("the frozen NeMo runtime is unavailable") from exc
    symbol_origins = {
        "SortformerEncLabelModel": _assert_symbol_origin(
            SortformerEncLabelModel, checkout, "SortformerEncLabelModel"
        ),
        "get_ats_targets": _assert_symbol_origin(get_ats_targets, checkout, "get_ats_targets"),
    }
    sortformer = SortformerEncLabelModel.restore_from(
        restore_path=str(checkpoint_path.resolve()), map_location=device
    )
    sortformer.streaming_mode = True
    sortformer.async_streaming = False
    for field, value in LOW_LATENCY_STREAMING.items():
        setattr(sortformer.sortformer_modules, field, value)
        if hasattr(sortformer._cfg.sortformer_modules, field):
            setattr(sortformer._cfg.sortformer_modules, field, value)
    sortformer.sortformer_modules._check_streaming_parameters()
    module_origins = _assert_loaded_nemo_origins(checkout)
    wrapped = TrainableSortformerPSEM(sortformer, get_ats_targets, checkout).to(device)
    graph = validate_streaming_graph(wrapped)
    return wrapped, {
        "checkpoint_sha256": NATIVE_SORTFORMER_CHECKPOINT_SHA256,
        "nemo_revision": head,
        "nemo_symbol_origins": symbol_origins,
        "nemo_loaded_module_origins": module_origins,
        "dependency_lock": lock,
        "dependency_lock_sha256": lock["sha256"],
        "model_graph": graph,
    }


class TrainableSortformerPSEM(nn.Module):
    def __init__(
        self,
        sortformer: nn.Module,
        get_ats_targets: Any,
        nemo_checkout: Path | None = None,
    ) -> None:
        super().__init__()
        self.sortformer = sortformer
        self.psem_head = PSEMHead()
        self.runtime_taps = nn.ModuleDict(
            {
                "final_temporal_hidden": nn.Identity(),
                "speaker_activity_logits": nn.Identity(),
            }
        )
        self._get_ats_targets = get_ats_targets
        self._nemo_checkout = nemo_checkout.resolve() if nemo_checkout is not None else None

    def _validate_nemo_runtime_origins(self) -> None:
        if self._nemo_checkout is not None:
            _assert_loaded_nemo_origins(self._nemo_checkout)

    def _infer_evidence(
        self, embeddings: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        modules = self.sortformer.sortformer_modules
        mask = modules.length_to_mask(lengths, embeddings.shape[1])
        hidden = self.sortformer.transformer_encoder(encoder_states=embeddings, encoder_mask=mask)
        activated = modules.dropout(F.relu(hidden))
        activated = modules.first_hidden_to_hidden(activated)
        activated = modules.dropout(F.relu(activated))
        logits = modules.single_hidden_to_spks(activated)
        probabilities = torch.sigmoid(logits) * mask.unsqueeze(-1)
        return hidden, logits, probabilities

    def _streaming_step(
        self,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor,
        streaming_state: Any,
        *,
        left_offset: int,
        right_offset: int,
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.sortformer
        modules = model.sortformer_modules
        chunk, chunk_lengths = model.encoder.pre_encode(
            x=processed_signal, lengths=processed_signal_length
        )
        if model.async_streaming:
            combined, combined_lengths = modules.concat_and_pad(
                [streaming_state.spkcache, streaming_state.fifo, chunk],
                [streaming_state.spkcache_lengths, streaming_state.fifo_lengths, chunk_lengths],
            )
        else:
            combined = modules.concat_embs(
                [streaming_state.spkcache, streaming_state.fifo, chunk],
                dim=1,
                device=model.device,
            )
            combined_lengths = (
                streaming_state.spkcache.shape[1] + streaming_state.fifo.shape[1] + chunk_lengths
            )
        encoded, encoded_lengths = model.frontend_encoder(
            processed_signal=combined,
            processed_signal_length=combined_lengths,
            bypass_pre_encode=True,
        )
        hidden, logits, probabilities = self._infer_evidence(encoded, encoded_lengths)
        probabilities = modules.apply_mask_to_preds(probabilities, encoded_lengths)
        lc = round(left_offset / model.encoder.subsampling_factor)
        rc = math.ceil(right_offset / model.encoder.subsampling_factor)
        prefix = streaming_state.spkcache.shape[1] + streaming_state.fifo.shape[1]
        current_count = chunk.shape[1] - lc - rc
        start = prefix + lc
        end = start + current_count
        aligned_logits = logits
        if streaming_state.spk_perm is not None:
            inverse = torch.stack(
                [torch.argsort(streaming_state.spk_perm[index]) for index in range(logits.shape[0])]
            )
            aligned_logits = torch.stack(
                [logits[index, :, inverse[index]] for index in range(logits.shape[0])]
            )
        current_hidden = hidden[:, start:end]
        current_logits = aligned_logits[:, start:end]
        if model.async_streaming:
            streaming_state, current_probabilities = modules.streaming_update_async(
                streaming_state=streaming_state,
                chunk=chunk,
                chunk_lengths=chunk_lengths,
                preds=probabilities,
                lc=lc,
                rc=rc,
            )
        else:
            streaming_state, current_probabilities = modules.streaming_update(
                streaming_state=streaming_state,
                chunk=chunk,
                preds=probabilities,
                lc=lc,
                rc=rc,
            )
        if current_hidden.shape[:2] != current_probabilities.shape[:2]:
            raise NeMoAdapterError("streaming hidden and posterior frame grids differ")
        if current_logits.shape != current_probabilities.shape:
            raise NeMoAdapterError("streaming activity logits and posterior geometry differ")
        return streaming_state, current_hidden, current_logits, current_probabilities

    def sortformer_evidence(
        self,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
        *,
        state_reset: torch.Tensor,
    ) -> SortformerEvidence:
        self._validate_nemo_runtime_origins()
        model = self.sortformer
        if (
            waveform.ndim != 2
            or waveform_lengths.shape != (waveform.shape[0],)
            or state_reset.ndim != 3
            or state_reset.shape[0] != waveform.shape[0]
            or state_reset.shape[2] != 1
            or state_reset.dtype != torch.bool
        ):
            raise NeMoAdapterError("waveform input geometry is invalid")
        processed, processed_lengths = model.process_signal(
            audio_signal=waveform, audio_signal_length=waveform_lengths
        )
        processed = processed[:, :, : processed_lengths.max()]
        state = model.sortformer_modules.init_streaming_state(
            batch_size=processed.shape[0],
            async_streaming=model.async_streaming,
            device=model.device,
        )
        offsets = torch.zeros((processed.shape[0],), dtype=torch.long, device=model.device)
        hidden_parts = []
        logit_parts = []
        probability_parts = []
        attention_modified = (
            model.training and random.random() < model.sortformer_modules.causal_attn_rate
        )
        with _temporary_causal_attention(model, attention_modified):
            loader = model.sortformer_modules.streaming_feat_loader(
                feat_seq=processed,
                feat_seq_length=processed_lengths,
                feat_seq_offset=offsets,
            )
            for _, chunk, lengths, left_offset, right_offset in loader:
                state, hidden, logits, probabilities = self._streaming_step(
                    chunk,
                    lengths,
                    state,
                    left_offset=left_offset,
                    right_offset=right_offset,
                )
                hidden_parts.append(hidden)
                logit_parts.append(logits)
                probability_parts.append(probabilities)
        if not hidden_parts:
            raise NeMoAdapterError("streaming graph produced no frames")
        hidden = torch.cat(hidden_parts, dim=1)
        logits = torch.cat(logit_parts, dim=1)
        probabilities = torch.cat(probability_parts, dim=1)
        expected_counts = torch.div(waveform_lengths, FRAME_SAMPLES, rounding_mode="floor")
        expected_frame_count = int(expected_counts[0])
        if (
            bool((waveform_lengths <= 0).any())
            or bool((waveform_lengths > waveform.shape[1]).any())
            or bool((waveform_lengths % FRAME_SAMPLES != 0).any())
            or int(expected_counts.min()) != int(expected_counts.max())
            or hidden.shape != (waveform.shape[0], expected_frame_count, 192)
            or logits.shape != (waveform.shape[0], expected_frame_count, 4)
            or probabilities.shape != logits.shape
        ):
            raise NeMoAdapterError("streaming output contains an incorrect final frame or tail")
        _validate_state_reset_lifecycle(
            state_reset,
            batch_size=waveform.shape[0],
            frame_count=expected_frame_count,
        )
        if not all(bool(torch.isfinite(value).all()) for value in (hidden, logits, probabilities)):
            raise NeMoAdapterError("streaming graph produced non-finite evidence")
        self._validate_nemo_runtime_origins()
        alive = torch.ones_like(probabilities, dtype=torch.int64)
        reset = state_reset.to(probabilities.device, probabilities.dtype)
        delay = torch.full_like(reset, MODEL_EVIDENCE_DELAY_SECONDS)
        return SortformerEvidence(
            probabilities=probabilities,
            activity_logits=logits,
            final_temporal_hidden=hidden,
            slot_alive=alive,
            state_reset=reset,
            evidence_delay_seconds=delay,
        )

    def psem_outputs(
        self,
        evidence: SortformerEvidence,
        oracle_anchor_slot_one_hot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = build_psem_features(
            evidence.final_temporal_hidden,
            evidence.activity_logits,
            evidence.slot_alive,
            oracle_anchor_slot_one_hot,
            evidence.state_reset,
            evidence.evidence_delay_seconds,
        )
        outputs, _ = self.psem_head(features)
        return outputs

    def runtime_canary_loss(
        self,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
    ) -> torch.Tensor:
        expected_frame_count = waveform.shape[1] // FRAME_SAMPLES
        state_reset = torch.zeros(
            (waveform.shape[0], expected_frame_count, 1),
            dtype=torch.bool,
            device=waveform.device,
        )
        state_reset[:, 0, 0] = True
        evidence = self.sortformer_evidence(
            waveform,
            waveform_lengths,
            state_reset=state_reset,
        )
        hidden = self.runtime_taps["final_temporal_hidden"](evidence.final_temporal_hidden)
        activity_logits = self.runtime_taps["speaker_activity_logits"](evidence.activity_logits)
        anchor = torch.zeros_like(evidence.activity_logits)
        anchor[:, :, 0] = 1
        features = build_psem_features(
            hidden,
            activity_logits,
            evidence.slot_alive,
            anchor,
            evidence.state_reset,
            evidence.evidence_delay_seconds,
        )
        outputs, _ = self.psem_head(features)
        return (
            outputs["anchor_present"].square().mean()
            + outputs["replacement_evidence"].square().mean()
        )

    def native_sortformer_loss(
        self,
        evidence: SortformerEvidence,
        arrival_order_targets: torch.Tensor,
        target_mask: torch.Tensor,
        sampling_roles: tuple[str, ...],
    ) -> Any:
        if arrival_order_targets.shape[-1] != 4:
            raise NeMoAdapterError("native supervision must contain four arrival-order slots")
        compact_probabilities, compact_arrival_targets, lengths = compact_valid_frames(
            evidence.probabilities,
            arrival_order_targets,
            target_mask,
        )
        targets = self._get_ats_targets(
            compact_arrival_targets.to(compact_probabilities.dtype),
            compact_probabilities,
            speaker_permutations=self.sortformer.speaker_permutations,
        )
        value = self.sortformer.loss(
            probs=compact_probabilities,
            labels=targets,
            target_lens=lengths,
        )
        self._validate_nemo_runtime_origins()
        return bind_native_sortformer_loss(
            value,
            sampling_roles=sampling_roles,
            kind=NATIVE_SORTFORMER_LOSS_KIND,
            origin=NATIVE_SORTFORMER_LOSS_ORIGIN,
            checkpoint_sha256=NATIVE_SORTFORMER_CHECKPOINT_SHA256,
        )


FIXED_RUNTIME_CANARY_METHODS = (
    ("runtime_canary_loss", TrainableSortformerPSEM.runtime_canary_loss),
    ("sortformer_evidence", TrainableSortformerPSEM.sortformer_evidence),
    ("_streaming_step", TrainableSortformerPSEM._streaming_step),
    ("_infer_evidence", TrainableSortformerPSEM._infer_evidence),
    ("_validate_nemo_runtime_origins", TrainableSortformerPSEM._validate_nemo_runtime_origins),
)
