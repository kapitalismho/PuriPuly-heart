from __future__ import annotations

"""FireRedChat-pvad adapter. LiveKit onset/offset smoothing (ExpFilter,
min_speech/min_silence durations, VADStream hysteresis) is bypassed: step()
returns the raw pvad.onnx probability with no averaging or debouncing."""

import hashlib
import os
from pathlib import Path

from experiments.psem_small_model_probe.adapter.protocol import (
    BindingError,
    StepOut,
    frame_bytes,
    validate_pcm16_chunk,
)

_VENDOR = Path(__file__).resolve().parent / "vendor"
_SUBFRAME_MS = 10
_SUBFRAME_BYTES = 160 * 2
_MEL_FLOATS = 1 * 80 * 15
_GRU_FLOATS = 2 * 1 * 256
_SPK_DIM = 192
_ONNX_NAME = "pvad.onnx"
_ECAPA_EMBEDDING = "embedding_model.ckpt"


def _zeros(n_floats: int) -> bytes:
    return b"\x00" * (n_floats * 4)


def _resolve_ecapa_dir(explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("FIRERED_ECAPA_DIR")
    if env:
        return Path(env)
    return _VENDOR / "spkrec-ecapa-voxceleb"


class FireRedAdapter:
    sample_rate_hz = 16000
    livekit_smoothing = False

    def __init__(
        self,
        frame_ms: int = 10,
        min_bind_ms: int = 1000,
        weights_dir: str | Path | None = None,
        ecapa_dir: str | Path | None = None,
    ) -> None:
        if frame_ms <= 0 or frame_ms % _SUBFRAME_MS != 0:
            raise ValueError(f"frame_ms must be a positive multiple of {_SUBFRAME_MS}")
        wdir = Path(weights_dir) if weights_dir is not None else _VENDOR
        self.onnx_path = wdir / _ONNX_NAME
        if not self.onnx_path.exists():
            raise FileNotFoundError(
                f"missing weights {self.onnx_path}; see adapter/vendor.json "
                "(artifact pvad_onnx) for the pinned URL"
            )
        self.ecapa_dir = _resolve_ecapa_dir(ecapa_dir)
        embedding = self.ecapa_dir / _ECAPA_EMBEDDING
        if not embedding.exists():
            raise FileNotFoundError(
                f"missing ECAPA weights {embedding}; fetch the receipt-only files "
                "in adapter/vendor.json (artifact spkrec_ecapa_voxceleb) into "
                "$FIRERED_ECAPA_DIR or adapter/vendor/spkrec-ecapa-voxceleb/"
            )
        self.frame_ms = frame_ms
        self.min_bind_ms = min_bind_ms
        self.onnx_sha = hashlib.sha256(self.onnx_path.read_bytes()).hexdigest()
        self.ecapa_sha = hashlib.sha256(embedding.read_bytes()).hexdigest()
        self._session = None
        self._reset_called = False
        self._bound = False
        self._source_time_ms = 0
        self.bind_span_hash: str | None = None
        self.frames: list[dict] = []
        self.reset()
        self._reset_called = False

    def reset(self) -> None:
        self._spkemb = _zeros(_SPK_DIM)
        self._mel = _zeros(_MEL_FLOATS)
        self._gru = _zeros(_GRU_FLOATS)
        self._reset_called = True
        self._bound = False
        self._source_time_ms = 0
        self.bind_span_hash = None
        self.frames = []

    def _enroll(self, reference_pcm16: bytes) -> None:
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise RuntimeError(
                "FireRedAdapter.bind() needs torch + speechbrain for ECAPA "
                f"enrollment (weights cached at {self.ecapa_dir}); "
                "install them or vendor a compatible enrollment path"
            ) from exc
        pcm = torch.frombuffer(bytearray(reference_pcm16), dtype=torch.int16)
        audio = (pcm.to(torch.float32) / 32768.0).unsqueeze(0)
        classifier = EncoderClassifier.from_hparams(
            source=str(self.ecapa_dir),
            savedir=str(self.ecapa_dir),
            run_opts={"device": "cpu"},
        )
        with torch.no_grad():
            embedding = classifier.encode_batch(audio)[0][0].detach()
            embedding = embedding / embedding.norm(p=2, dim=0, keepdim=True)
        self._spkemb = embedding.cpu().numpy().astype("<f4").tobytes()

    def bind(self, reference_pcm16: bytes) -> None:
        if not self._reset_called:
            raise RuntimeError("bind() requires reset() first")
        unit = frame_bytes(self.frame_ms, self.sample_rate_hz)
        if len(reference_pcm16) == 0 or len(reference_pcm16) < self.min_bind_ms * 16 * 2:
            raise BindingError(
                f"reference span too short: {len(reference_pcm16)} bytes "
                f"(need >= {self.min_bind_ms} ms mono 16 kHz int16 LE)"
            )
        if len(reference_pcm16) % unit != 0:
            raise ValueError("reference span must be a frame multiple")
        self.bind_span_hash = hashlib.sha256(reference_pcm16).hexdigest()
        self._enroll(reference_pcm16)
        self._bound = True

    def _session_or_raise(self):
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "FireRedAdapter.step() needs onnxruntime (CPU) to run "
                    f"{self.onnx_path.name}; install it, weights are vendored"
                ) from exc
            self._session = ort.InferenceSession(
                str(self.onnx_path), providers=["CPUExecutionProvider"]
            )
        return self._session

    def _raw_prob(self, subframe: bytes) -> float:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "FireRedAdapter.step() needs numpy to pack onnx inputs"
            ) from exc
        session = self._session_or_raise()
        pcm = np.frombuffer(subframe, dtype="<i2").astype(np.float32) / 32768.0
        prob, mel, gru = session.run(
            None,
            {
                "input_audio": pcm.reshape(1, 160),
                "spkemb": np.frombuffer(self._spkemb, dtype="<f4").reshape(1, _SPK_DIM),
                "mel_buffer": np.frombuffer(self._mel, dtype="<f4").reshape(1, 80, 15),
                "gru_buffer": np.frombuffer(self._gru, dtype="<f4").reshape(2, 1, 256),
            },
        )[1:4]
        raw = float(np.asarray(prob).reshape(-1)[0])
        assert 0.0 <= raw <= 1.0
        self._mel = np.asarray(mel, dtype=np.float32).tobytes()
        self._gru = np.asarray(gru, dtype=np.float32).tobytes()
        return raw

    def step(self, pcm16_chunk: bytes) -> StepOut:
        if not self._bound:
            raise RuntimeError("step() requires bind() first")
        n = validate_pcm16_chunk(
            pcm16_chunk, frame_ms=self.frame_ms, sample_rate_hz=self.sample_rate_hz
        )
        prev = self._source_time_ms
        out = None
        for i in range(n * (self.frame_ms // _SUBFRAME_MS)):
            raw = self._raw_prob(
                pcm16_chunk[i * _SUBFRAME_BYTES:(i + 1) * _SUBFRAME_BYTES]
            )
            t = prev + (i + 1) * _SUBFRAME_MS
            aux = {
                "ecapa_dim": _SPK_DIM,
                "mel_state_hash": hashlib.sha256(self._mel).hexdigest()[:16],
            }
            self.frames.append(
                {"source_time_ms": t, "speech": None, "anchor": raw, "aux": aux}
            )
            out = StepOut(speech=None, anchor=raw, aux=aux, source_time_ms=t)
        assert out is not None and out.source_time_ms > prev
        self._source_time_ms = out.source_time_ms
        return out

    def episode_header(self) -> dict:
        return {
            "model": "fireredchat-pvad",
            "model_sha": self.onnx_sha,
            "onnx_sha": self.onnx_sha,
            "ecapa_sha": self.ecapa_sha,
            "reset_ok": self._reset_called,
            "bind_span_hash": self.bind_span_hash,
        }
