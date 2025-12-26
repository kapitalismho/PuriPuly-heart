from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from puripuly_heart.core.vad.silero import SileroVadOnnx


class _NodeArg:
    def __init__(self, name: str, *, type: str, shape: list[int | None]):
        self.name = name
        self.type = type
        self.shape = shape


class _FakeSession:
    def __init__(self, _path: str, *, providers: list[str]):
        self.providers = providers
        self.calls: list[dict[str, object]] = []

        self._inputs = [
            _NodeArg("input", type="tensor(float)", shape=[1, 512]),
            _NodeArg("sr", type="tensor(int64)", shape=[1]),
            _NodeArg("h", type="tensor(float)", shape=[2, 1, 64]),
            _NodeArg("c", type="tensor(float)", shape=[2, 1, 64]),
        ]
        self._outputs = [
            _NodeArg("output", type="tensor(float)", shape=[1, 1]),
            _NodeArg("hn", type="tensor(float)", shape=[2, 1, 64]),
            _NodeArg("cn", type="tensor(float)", shape=[2, 1, 64]),
        ]

    def get_inputs(self):
        return list(self._inputs)

    def get_outputs(self):
        return list(self._outputs)

    def run(self, _output_names, feed: dict[str, object]):
        self.calls.append(feed)

        h = np.asarray(feed["h"], dtype=np.float32)
        c = np.asarray(feed["c"], dtype=np.float32)
        call_n = len(self.calls)

        if call_n == 1:
            assert np.all(h == 0.0)
            assert np.all(c == 0.0)
            prob = 0.7
        elif call_n == 2:
            assert np.all(h == 1.0)
            assert np.all(c == 1.0)
            prob = 0.2
        else:
            assert np.all(h == 0.0)
            assert np.all(c == 0.0)
            prob = 0.5

        hn = h + 1.0
        cn = c + 1.0
        return [
            np.asarray([[prob]], dtype=np.float32),
            hn,
            cn,
        ]


def test_silero_vad_onnx_inference_and_reset(tmp_path, monkeypatch):
    fake_ort = ModuleType("onnxruntime")
    fake_ort.InferenceSession = _FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    model_path = tmp_path / "silero.onnx"
    model_path.write_bytes(b"")

    vad = SileroVadOnnx(model_path=model_path)

    p1 = vad.speech_probability(np.zeros((512,), dtype=np.float32), sample_rate_hz=16000)
    p2 = vad.speech_probability(np.zeros((512,), dtype=np.float32), sample_rate_hz=16000)
    assert p1 == pytest.approx(0.7)
    assert p2 == pytest.approx(0.2)

    vad.reset()
    p3 = vad.speech_probability(np.zeros((512,), dtype=np.float32), sample_rate_hz=16000)
    assert p3 == pytest.approx(0.5)
