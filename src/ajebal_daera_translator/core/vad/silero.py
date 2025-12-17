from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class SileroVadOnnx:
    model_path: Path

    def __post_init__(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)

        try:
            import onnxruntime as ort  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "onnxruntime is required for SileroVadOnnx; install with `pip install onnxruntime`"
            ) from exc

        self._ort = ort
        self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])

    def reset(self) -> None:
        # Actual Silero VAD state management is implemented when the ONNX IO contract is finalized.
        return

    def speech_probability(self, samples: np.ndarray, *, sample_rate_hz: int) -> float:
        raise NotImplementedError(
            "SileroVadOnnx inference is not wired yet; use a FakeVadEngine for now."
        )

