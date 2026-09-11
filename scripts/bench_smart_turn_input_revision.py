from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from puripuly_heart.core.audio.smart_turn import (
    SMART_TURN_INPUT_REVISION,
    SmartTurnOnnxInference,
    prepare_smart_turn_audio,
)
from puripuly_heart.core.audio.smart_turn_features import compute_whisper_log_mel_features

_REFERENCE_URL = (
    "https://raw.githubusercontent.com/kapitalismho/PuriPuly-heart/"
    "8dd248b8f73556ac32d24c00223b4b413d4aca98/src/puripuly_heart/core/vad/smart_turn_features.py"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(value).tobytes())


def _fixture() -> np.ndarray:
    return (
        0.25 * np.sin(2.0 * np.pi * 220.0 * np.arange(16000, dtype=np.float64) / 16000.0)
    ).astype(np.float32)


def _reference_feature_function():
    with urllib.request.urlopen(_REFERENCE_URL, timeout=30.0) as response:
        source = response.read()
    namespace: dict[str, object] = {}
    exec(compile(source, _REFERENCE_URL, "exec"), namespace)
    return namespace["compute_whisper_log_mel_features"], _sha256_bytes(source)


async def _verify(model_path: Path, repeats: int) -> dict[str, object]:
    model_digest = _sha256_bytes(model_path.read_bytes())
    raw = _fixture()
    prepared = prepare_smart_turn_audio(raw, sample_rate_hz=16000)
    current_features = compute_whisper_log_mel_features(prepared)
    reference_fn, reference_source_digest = _reference_feature_function()
    reference_features = reference_fn(prepared)
    model = SmartTurnOnnxInference(model_path)
    scores: list[float] = []
    inference_ms: list[float] = []
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            scores.append(await model.predict(raw, sample_rate_hz=16000))
            inference_ms.append((time.perf_counter() - started) * 1000.0)
    finally:
        model.close()
    difference = np.abs(current_features - reference_features)
    return {
        "input_revision": SMART_TURN_INPUT_REVISION,
        "reference_url": _REFERENCE_URL,
        "reference_source_sha256": reference_source_digest,
        "model_path": str(model_path),
        "model_sha256": model_digest,
        "fixture": {
            "description": "float32 0.25-amplitude 220 Hz sine",
            "sample_rate_hz": 16000,
            "samples": int(raw.size),
            "raw_sha256": _array_sha256(raw),
            "prepared_sha256": _array_sha256(prepared),
            "features_sha256": _array_sha256(current_features),
            "features_shape": list(current_features.shape),
        },
        "feature_parity": {
            "array_equal": bool(np.array_equal(current_features, reference_features)),
            "max_abs_difference": float(difference.max(initial=0.0)),
        },
        "scores": scores,
        "inference_ms": inference_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    print(json.dumps(asyncio.run(_verify(args.model, args.repeats)), indent=2))


if __name__ == "__main__":
    main()
