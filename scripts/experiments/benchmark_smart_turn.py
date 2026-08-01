from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from puripuly_heart.core.vad.smart_turn import SmartTurnOnnxInference


async def _benchmark(
    model_path: Path,
    audio_paths: tuple[Path, ...],
    repeats: int,
) -> dict[str, object]:
    model = SmartTurnOnnxInference(model_path)
    records: list[dict[str, object]] = []
    for audio_path in audio_paths:
        audio = np.asarray(np.load(audio_path), dtype=np.float32).reshape(-1)
        latencies: list[float] = []
        scores: list[float] = []
        for _ in range(repeats):
            started_at = time.perf_counter()
            prediction = await model.predict(audio, sample_rate_hz=16000)
            latencies.append((time.perf_counter() - started_at) * 1000.0)
            scores.append(prediction.score)
        records.append(
            {
                "audio": str(audio_path),
                "samples": int(audio.size),
                "duration_ms": float(audio.size * 1000.0 / 16000.0),
                "scores": scores,
                "inference_ms": {
                    "mean": float(np.mean(latencies)),
                    "p50": float(np.percentile(latencies, 50)),
                    "p95": float(np.percentile(latencies, 95)),
                },
            }
        )
    return {
        "model": str(model_path),
        "repeats": repeats,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    print(
        json.dumps(
            asyncio.run(_benchmark(args.model, tuple(args.audio), args.repeats)),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
