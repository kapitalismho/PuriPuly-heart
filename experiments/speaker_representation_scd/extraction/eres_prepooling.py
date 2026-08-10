from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.speaker_representation_scd.extraction.common import ExtractionBatch
from experiments.speaker_representation_scd.provenance import load_json, verify_file_identity

TAP_MODULES = {
    "S1": "layer1",
    "S2": "layer2",
    "S3": "layer3",
    "S4": "layer4",
    "FUSED": "fuse34",
}


class ERes2NetV2PrepoolExtractor:
    def __init__(
        self,
        checkpoint_root: Path,
        source_root: Path,
        source_registry_path: Path,
        *,
        threads: int = 8,
    ) -> None:
        import torch

        self.model_id = "eres2netv2-standard-prepool"
        self.checkpoint_root = checkpoint_root.resolve()
        self.source_root = source_root.resolve()
        registry = load_json(source_registry_path)
        model_contract = registry["eres2netv2"]
        errors: list[str] = []
        for row in model_contract["source_files"]:
            errors.extend(
                verify_file_identity(self.source_root / row["path"], row["sha256"], None)
            )
        license_row = model_contract["source_license"]
        errors.extend(
            verify_file_identity(
                self.source_root / license_row["path"], license_row["sha256"], None
            )
        )
        checkpoint_row = model_contract["checkpoint_file"]
        errors.extend(
            verify_file_identity(
                self.checkpoint_root / checkpoint_row["path"],
                checkpoint_row["sha256"],
                checkpoint_row["size_bytes"],
            )
        )
        if errors:
            raise ValueError("; ".join(errors))
        source_value = str(self.source_root)
        if source_value not in sys.path:
            sys.path.insert(0, source_value)
        model_module = importlib.import_module("speakerlab.models.eres2net.ERes2NetV2")
        fusion_module = importlib.import_module("speakerlab.models.eres2net.fusion")
        pooling_module = importlib.import_module("speakerlab.models.eres2net.pooling_layers")
        processor_module = importlib.import_module("speakerlab.process.processor")
        augmentation_module = importlib.import_module("speakerlab.process.augmentation")
        for module in (
            model_module,
            fusion_module,
            pooling_module,
            processor_module,
            augmentation_module,
        ):
            module_path = Path(module.__file__).resolve()
            if self.source_root not in module_path.parents:
                raise RuntimeError(f"ERes module was imported from an unverified root: {module_path}")
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.use_deterministic_algorithms(True)
        self.model = model_module.ERes2NetV2(
            feat_dim=80,
            embedding_size=192,
            m_channels=64,
        )
        checkpoint = torch.load(
            self.checkpoint_root / checkpoint_row["path"],
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise RuntimeError("ERes checkpoint is not a state dictionary")
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()
        self.frontend = processor_module.FBank(80, sample_rate=16000, mean_nor=True)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    def _features(self, waveforms: list[np.ndarray]):
        import torch

        values = []
        for waveform in waveforms:
            tensor = torch.from_numpy(waveform)
            values.append(self.frontend(tensor, dither=0))
        shapes = {tuple(value.shape) for value in values}
        if len(shapes) != 1:
            raise ValueError("ERes batches require equal frontend shapes")
        return torch.stack(values)

    def extract(
        self,
        waveforms: Iterable[np.ndarray],
        observed_source_samples: Iterable[int],
        tap_ids: Iterable[str] = tuple(TAP_MODULES),
    ) -> ExtractionBatch:
        import torch

        rows = [np.ascontiguousarray(row, dtype=np.float32) for row in waveforms]
        if not rows or any(row.ndim != 1 for row in rows):
            raise ValueError("at least one one-dimensional waveform is required")
        if len({row.shape[0] for row in rows}) != 1:
            raise ValueError("ERes batches require equal-length waveforms")
        selected = tuple(tap_ids)
        unknown = set(selected) - set(TAP_MODULES)
        if unknown:
            raise ValueError(f"unsupported ERes taps: {sorted(unknown)}")
        captured: dict[str, Any] = {}
        handles = []
        for tap_id in selected:
            module = getattr(self.model, TAP_MODULES[tap_id])

            def capture(_module, _inputs, output, *, name=tap_id):
                captured[name] = output

            handles.append(module.register_forward_hook(capture))
        features = self._features(rows)
        try:
            with torch.inference_mode():
                embedding = self.model(features)
        finally:
            for handle in handles:
                handle.remove()
        layers: dict[str, np.ndarray] = {}
        valid: dict[str, np.ndarray] = {}
        for tap_id in selected:
            value = captured[tap_id]
            flat = value.permute(0, 3, 1, 2).reshape(value.shape[0], value.shape[3], -1)
            array = flat.detach().cpu().to(torch.float32).numpy()
            layers[tap_id] = np.ascontiguousarray(array)
            valid[tap_id] = np.full(array.shape[0], array.shape[1], dtype=np.int64)
        observed = np.asarray(tuple(observed_source_samples), dtype=np.int64)
        if observed.shape != (len(rows),):
            raise ValueError("observed_source_samples must have one entry per waveform")
        return ExtractionBatch(
            model_id=self.model_id,
            layers=layers,
            valid_lengths=valid,
            observed_source_samples=observed,
            official_embedding=embedding.detach().cpu().to(torch.float32).numpy(),
        )

    def reconstruct_embedding(self, fused: np.ndarray) -> np.ndarray:
        import torch

        if fused.ndim != 4:
            raise ValueError("fused tensor must have shape batch,channel,frequency,time")
        tensor = torch.from_numpy(np.ascontiguousarray(fused, dtype=np.float32))
        with torch.inference_mode():
            stats = self.model.pool(tensor)
            embedding = self.model.seg_1(stats)
            if self.model.two_emb_layer:
                value = torch.nn.functional.relu(embedding)
                value = self.model.seg_bn_1(value)
                embedding = self.model.seg_2(value)
        return embedding.detach().cpu().to(torch.float32).numpy()
