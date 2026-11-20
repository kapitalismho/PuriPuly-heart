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
    _sr_input_is_scalar: bool = field(init=False, default=False)
    _state_input_names: tuple[str, ...] = field(init=False, default=())
    _state_output_names: dict[str, str] = field(init=False, default_factory=dict)
    _prob_output_name: str = field(init=False)
    _output_names: tuple[str, ...] = field(init=False, default=())
    _state: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    _initial_state: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    _audio_input: np.ndarray | None = field(init=False, default=None, repr=False)
    _sample_rate_input: np.ndarray | None = field(init=False, default=None, repr=False)
    _last_sr: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)

        import onnxruntime as ort  # type: ignore

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._configure_io()
        self.reset()

    def reset(self) -> None:
        self._state = {name: value.copy() for name, value in self._initial_state.items()}
        self._audio_input = None
        self._sample_rate_input = None
        self._last_sr = 0

    def speech_probability(self, samples: np.ndarray, *, sample_rate_hz: int) -> float:
        if sample_rate_hz not in (8000, 16000):
            raise ValueError("Silero VAD streaming supports only 8000 or 16000 Hz")

        chunk = np.asarray(samples, dtype=np.float32)
        if chunk.ndim == 1:
            batch_chunk = chunk.reshape(1, -1)
        elif chunk.ndim == 2 and chunk.shape[0] == 1:
            batch_chunk = chunk
        else:
            raise ValueError(f"Too many dimensions for input audio chunk {chunk.ndim}")

        expected_chunk_samples = self._chunk_samples_for(sample_rate_hz)
        if batch_chunk.shape[-1] != expected_chunk_samples:
            raise ValueError(
                "Provided number of samples is "
                f"{batch_chunk.shape[-1]} "
                "(Supported values: 256 for 8000 sample rate, 512 for 16000)"
            )

        context_samples = self._context_samples_for(sample_rate_hz)
        if self._last_sr and self._last_sr != sample_rate_hz:
            self.reset()
        if self._audio_input is None:
            self._audio_input = np.empty((1, context_samples + expected_chunk_samples), np.float32)
            self._audio_input[:, :context_samples] = 0.0
            if self._sr_input_name is not None:
                self._sample_rate_input = np.asarray(
                    sample_rate_hz if self._sr_input_is_scalar else [sample_rate_hz],
                    dtype=np.int64,
                )
            self._last_sr = sample_rate_hz
        audio_input = self._audio_input
        audio_input[:, context_samples:] = batch_chunk
        feed: dict[str, Any] = {self._audio_input_name: audio_input}
        if self._sr_input_name is not None:
            feed[self._sr_input_name] = self._sample_rate_input
        for name in self._state_input_names:
            feed[name] = self._state[name]

        outputs = self._session.run(None, feed)
        by_name = dict(zip(self._output_names, outputs, strict=True))

        prob_raw = by_name[self._prob_output_name]
        prob = float(np.asarray(prob_raw, dtype=np.float32).reshape(-1)[0])

        for input_name, output_name in self._state_output_names.items():
            if output_name in by_name:
                self._state[input_name] = np.asarray(by_name[output_name], dtype=np.float32)

        audio_input[:, :context_samples] = audio_input[:, -context_samples:]

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
        if self._sr_input_name is not None:
            sr_shape = getattr(inputs[self._sr_input_name], "shape", None) or []
            self._sr_input_is_scalar = len(sr_shape) == 0

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

    @staticmethod
    def _chunk_samples_for(sample_rate_hz: int) -> int:
        return 512 if sample_rate_hz == 16000 else 256

    @staticmethod
    def _context_samples_for(sample_rate_hz: int) -> int:
        return 64 if sample_rate_hz == 16000 else 32
