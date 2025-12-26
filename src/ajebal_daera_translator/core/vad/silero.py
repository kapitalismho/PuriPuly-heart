from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class SileroVadOnnx:
    model_path: Path
    _session: Any = field(init=False, repr=False)
    _audio_input_name: str = field(init=False)
    _sr_input_name: str | None = field(init=False, default=None)
    _state_input_names: tuple[str, ...] = field(init=False, default=())
    _state_output_names: dict[str, str] = field(init=False, default_factory=dict)
    _prob_output_name: str = field(init=False)
    _output_names: tuple[str, ...] = field(init=False, default=())
    _expected_chunk_samples: int | None = field(init=False, default=None)
    _state: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    _initial_state: dict[str, np.ndarray] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)

        import onnxruntime as ort  # type: ignore

        self._session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self._configure_io()
        self.reset()

    def reset(self) -> None:
        self._state = {name: value.copy() for name, value in self._initial_state.items()}

    def speech_probability(self, samples: np.ndarray, *, sample_rate_hz: int) -> float:
        if sample_rate_hz not in (8000, 16000):
            raise ValueError("Silero VAD streaming supports only 8000 or 16000 Hz")

        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self._expected_chunk_samples is not None and chunk.size != self._expected_chunk_samples:
            raise ValueError(f"Expected {self._expected_chunk_samples} samples, got {chunk.size}")

        feed: dict[str, Any] = {self._audio_input_name: chunk.reshape(1, -1)}
        if self._sr_input_name is not None:
            feed[self._sr_input_name] = np.asarray([sample_rate_hz], dtype=np.int64)
        for name in self._state_input_names:
            feed[name] = self._state[name]

        outputs = self._session.run(None, feed)
        by_name = dict(zip(self._output_names, outputs, strict=True))

        prob_raw = by_name[self._prob_output_name]
        prob = float(np.asarray(prob_raw, dtype=np.float32).reshape(-1)[0])

        for input_name, output_name in self._state_output_names.items():
            if output_name in by_name:
                self._state[input_name] = np.asarray(by_name[output_name], dtype=np.float32)

        return prob

    def _configure_io(self) -> None:
        inputs = {i.name: i for i in self._session.get_inputs()}
        outputs = [o.name for o in self._session.get_outputs()]

        self._output_names = tuple(outputs)

        if "input" in inputs:
            self._audio_input_name = "input"
        elif "x" in inputs:
            self._audio_input_name = "x"
        else:
            float_inputs = [i for i in inputs.values() if "float" in str(getattr(i, "type", ""))]
            if not float_inputs:
                raise ValueError("Silero VAD ONNX model has no float inputs")
            float_inputs.sort(key=lambda i: len(getattr(i, "shape", []) or []))
            self._audio_input_name = float_inputs[0].name

        if "sr" in inputs:
            self._sr_input_name = "sr"
        elif "sample_rate" in inputs:
            self._sr_input_name = "sample_rate"

        state_inputs: list[str] = []
        # Silero VAD v5+ uses 'state' as a single input
        if "state" in inputs:
            state_inputs.append("state")
        else:
            # Older versions use 'h' and 'c' separately
            for name in ("h", "c"):
                if name in inputs:
                    state_inputs.append(name)
        self._state_input_names = tuple(state_inputs)

        audio_shape = getattr(inputs[self._audio_input_name], "shape", None) or []
        if isinstance(audio_shape, (list, tuple)) and audio_shape:
            last_dim = audio_shape[-1]
            if isinstance(last_dim, int) and last_dim > 0:
                self._expected_chunk_samples = int(last_dim)

        output_set = set(outputs)
        if "output" in output_set:
            self._prob_output_name = "output"
        elif "prob" in output_set:
            self._prob_output_name = "prob"
        else:
            self._prob_output_name = outputs[0]

        # Silero VAD v5+ output mapping
        if "state" in self._state_input_names:
            # v5 uses 'stateN' as output for 'state' input
            for out_name in output_set:
                if out_name.startswith("state") and out_name != "state":
                    self._state_output_names["state"] = out_name
                    break
            # Fallback: if 'stateN' not found, try 'state' itself
            if "state" not in self._state_output_names and "state" in output_set:
                self._state_output_names["state"] = "state"

        if "h" in self._state_input_names:
            if "hn" in output_set:
                self._state_output_names["h"] = "hn"
            elif "h" in output_set:
                self._state_output_names["h"] = "h"

        if "c" in self._state_input_names:
            if "cn" in output_set:
                self._state_output_names["c"] = "cn"
            elif "c" in output_set:
                self._state_output_names["c"] = "c"

        def _state_shape(name: str) -> tuple[int, ...]:
            raw_shape = getattr(inputs[name], "shape", None) or []
            dims: list[int] = []
            for dim in raw_shape:
                if isinstance(dim, int) and dim > 0:
                    dims.append(dim)
                else:
                    dims.append(1)
            return tuple(dims) or (1,)

        self._initial_state = {}
        for name in self._state_input_names:
            self._initial_state[name] = np.zeros(_state_shape(name), dtype=np.float32)
