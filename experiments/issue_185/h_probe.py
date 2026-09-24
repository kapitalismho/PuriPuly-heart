from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import tracemalloc

import numpy as np

from puripuly_heart.core.audio.format import float32_to_pcm16le_bytes, pcm16le_bytes_to_float32


def measure(work):
    measurements = []
    for _ in range(100):
        tracemalloc.start()
        start = time.perf_counter_ns()
        samples = work()
        elapsed = time.perf_counter_ns() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        measurements.append((elapsed / 1000, peak))
        yield samples, measurements[-1]


def main() -> None:
    source = np.tile(
        np.array([-1.0, -0.9, -0.5, -0.00003, 0.0, 0.00003, 0.5, 0.9, 1.0], dtype=np.float32), 1778
    )
    pcm = float32_to_pcm16le_bytes(source)
    expected = pcm16le_bytes_to_float32(pcm)
    before = list(measure(lambda: pcm16le_bytes_to_float32(pcm).copy()))
    after = list(measure(lambda: pcm16le_bytes_to_float32(pcm)))
    for samples, _ in (*before, *after):
        np.testing.assert_array_equal(samples, expected)
    all_pcm = np.arange(-32768, 32768, dtype=np.int32).astype("<i2").tobytes()
    old_wav = (
        np.rint(np.clip(pcm16le_bytes_to_float32(all_pcm), -1, 1) * 32767).astype("<i2").tobytes()
    )
    routed_wav = (
        np.rint(
            np.clip(pcm16le_bytes_to_float32(b"".join([all_pcm[:50000], all_pcm[50000:]])), -1, 1)
            * 32767
        )
        .astype("<i2")
        .tobytes()
    )
    assert routed_wav == old_wav
    print(
        json.dumps(
            {
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "profile": "100 Qwen 16002-sample PCM16 conversions; GPU all 65536 PCM16 codes",
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "platform": platform.platform(),
                "qwen_before_median_us": float(np.median([row[1][0] for row in before])),
                "qwen_after_median_us": float(np.median([row[1][0] for row in after])),
                "qwen_before_peak_bytes": int(np.median([row[1][1] for row in before])),
                "qwen_after_peak_bytes": int(np.median([row[1][1] for row in after])),
                "gpu_quantized_codes_equal": 65536,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
