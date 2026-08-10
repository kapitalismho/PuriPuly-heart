from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from experiments.speaker_representation_scd.extraction.common import ExtractionBatch

MODEL_CLASS_NAMES = {
    "mhubert-147": "HubertModel",
    "wavlm-base-plus": "WavLMModel",
    "unispeech-sat-base-plus": "UniSpeechSatModel",
}
LAYER_IDS = ("L1", "L3", "L6", "L9", "L12")


class SSLExtractor:
    def __init__(self, model_id: str, model_root: Path, *, threads: int = 8) -> None:
        import torch
        import transformers

        if model_id not in MODEL_CLASS_NAMES:
            raise ValueError(f"unsupported SSL model: {model_id}")
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.use_deterministic_algorithms(True)
        loader = getattr(transformers, MODEL_CLASS_NAMES[model_id])
        self.model_id = model_id
        self.model_root = model_root.resolve()
        self.processor = transformers.AutoFeatureExtractor.from_pretrained(
            str(self.model_root), local_files_only=True, trust_remote_code=False
        )
        self.model = loader.from_pretrained(
            str(self.model_root),
            local_files_only=True,
            trust_remote_code=False,
            weights_only=True,
        )
        self.model.eval()
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    def expected_output_length(self, input_samples: int) -> int:
        import torch

        length = self.model._get_feat_extract_output_lengths(
            torch.tensor([input_samples], dtype=torch.long)
        )
        return int(length.item())

    def extract(
        self,
        waveforms: Iterable[np.ndarray],
        observed_source_samples: Iterable[int],
        layer_ids: Iterable[str] = LAYER_IDS,
    ) -> ExtractionBatch:
        import torch

        rows = [np.ascontiguousarray(row, dtype=np.float32) for row in waveforms]
        if not rows:
            raise ValueError("at least one waveform is required")
        lengths = {row.shape[0] for row in rows}
        if len(lengths) != 1 or any(row.ndim != 1 for row in rows):
            raise ValueError("SSL batches require equal-length one-dimensional waveforms")
        selected = tuple(layer_ids)
        indices = []
        for layer_id in selected:
            if layer_id not in LAYER_IDS:
                raise ValueError(f"unsupported SSL layer: {layer_id}")
            indices.append(int(layer_id[1:]))
        encoded = self.processor(
            rows,
            sampling_rate=16000,
            padding=False,
            return_tensors="pt",
        )
        arguments = {"input_values": encoded["input_values"]}
        if "attention_mask" in encoded:
            arguments["attention_mask"] = encoded["attention_mask"]
        with torch.inference_mode():
            output = self.model(
                **arguments,
                output_hidden_states=True,
                return_dict=True,
            )
        if output.hidden_states is None or len(output.hidden_states) != 13:
            raise RuntimeError("SSL extractor did not return L0 plus twelve block outputs")
        output_length = self.expected_output_length(next(iter(lengths)))
        features: dict[str, np.ndarray] = {}
        valid: dict[str, np.ndarray] = {}
        for layer_id, index in zip(selected, indices, strict=True):
            values = output.hidden_states[index].detach().cpu().to(torch.float32).numpy()
            if values.shape[1] != output_length:
                raise RuntimeError(
                    f"SSL output length mismatch for {layer_id}: {values.shape[1]} != {output_length}"
                )
            features[layer_id] = np.ascontiguousarray(values)
            valid[layer_id] = np.full(values.shape[0], output_length, dtype=np.int64)
        observed = np.asarray(tuple(observed_source_samples), dtype=np.int64)
        if observed.shape != (len(rows),):
            raise ValueError("observed_source_samples must have one entry per waveform")
        return ExtractionBatch(
            model_id=self.model_id,
            layers=features,
            valid_lengths=valid,
            observed_source_samples=observed,
        )
