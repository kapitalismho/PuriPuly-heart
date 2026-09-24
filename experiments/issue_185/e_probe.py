"""Opt-in SoXR 1.1.0 sample-frontier grid and actual mapped-normalizer/VAD exercise."""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soxr

from puripuly_heart.core.audio.format import AudioCaptureDiscontinuity, AudioCaptureSpan
from puripuly_heart.core.audio.streaming_resampler import (
    CaptureMappedStreamingResampler,
    MonoFirstStreamingResampler,
)
from puripuly_heart.core.runtime.audio_vad_loop import _capture_prefix
from puripuly_heart.core.vad.gating import VadGating


def dist(seq):
    seq = sorted(seq)
    return (
        {
            "n": len(seq),
            "min": seq[0],
            "p50": statistics.median(seq),
            "p95": seq[int((len(seq) - 1) * 0.95)],
            "max": seq[-1],
        }
        if seq
        else {"n": 0}
    )


def fixture(rate, seconds=3):
    t = np.arange(round(rate * seconds), dtype=np.float64) / rate
    envelope = np.where(((t >= 0.2) & (t < 1.2)) | ((t >= 1.5) & (t < 2.4)), 0.17, 0)
    return (envelope * (np.sin(2 * np.pi * 175 * t) + 0.3 * np.sin(2 * np.pi * 350 * t))).astype(
        np.float32
    )


def grid(rate, block, quality):
    input_audio = np.zeros(3 * rate, dtype=np.float32)
    stream = (
        None
        if rate == 16000
        else soxr.ResampleStream(rate, 16000, 1, dtype="float32", quality=quality)
    )
    count = 0
    pending = []
    vad = []
    emitted = []
    sizes = []
    first = None
    cpu = []
    pre = 0
    for start in range(0, input_audio.size, block):
        input_chunk = input_audio[start : start + block]
        t0 = time.perf_counter_ns()
        output = input_chunk if stream is None else stream.resample_chunk(input_chunk)
        cpu.append((time.perf_counter_ns() - t0) / 1000)
        count += input_chunk.size
        pre += output.size
        sizes.append(int(output.size))
        if first is None and output.size:
            first = count / rate * 1000
        pending.append((stream.delay() / 16000 * 1000) if stream is not None else 0.0)
        age = count / rate * 1000 - (pre // 512 * 512) / 16000 * 1000
        vad.append(age)
        if output.size:
            emitted.append(age)
    tail = (
        0
        if stream is None
        else int(stream.resample_chunk(np.empty(0, dtype=np.float32), last=True).size)
    )
    expected = round(input_audio.size * 16000 / rate)
    assert pre + tail == expected, (rate, block, quality, pre, tail, expected)
    return {
        "rate": rate,
        "block": block,
        "quality": quality if stream is not None else "bypass",
        "callbacks": len(sizes),
        "first_output_input_frontier_ms": first,
        "pending_ms": dist(pending),
        "vad_frontier_age_ms": dist(vad),
        "vad_frontier_age_emitting_ms": dist(emitted),
        "output_sizes": dict((str(n), sizes.count(n)) for n in sorted(set(sizes))),
        "resample_call_us": dist(cpu),
        "pre_flush_samples": pre,
        "flush_tail_samples": tail,
        "total_samples": pre + tail,
        "expected_samples": expected,
    }


class AmplitudeEngine:
    """Deterministic interface probe only; never substitute for Silero speech-quality evidence."""

    def speech_probability(self, samples, *, sample_rate_hz):
        return 1.0 if np.max(np.abs(samples)) > 0.12 else 0.0

    def reset(self):
        pass


def mapped_path(rate, block):
    signal = fixture(rate)
    mapper = CaptureMappedStreamingResampler(rate)
    vad = VadGating(AmplitudeEngine(), sample_rate_hz=16000, ring_buffer_ms=500, hangover_ms=128)
    in_count = 0
    consumed = 0
    buffer = np.empty(0, dtype=np.float32)
    spans = []
    observations = []
    first = None
    pending = []
    capture_buffer = []
    for seq, start in enumerate(range(0, signal.size, block)):
        chunk = signal[start : start + block]
        capture = AudioCaptureSpan(
            0, seq, rate, start, start + chunk.size, start / rate, (start + chunk.size) / rate
        )
        output, mapped, discarded = mapper.process(chunk, capture)
        assert not discarded
        if mapped is not None:
            assert mapped.normalized_start_sample == consumed + buffer.size
            spans.append(mapped)
            capture_buffer.append(mapped)
        buffer = np.concatenate((buffer, output))
        in_count += chunk.size
        if first is None and output.size:
            first = in_count / rate * 1000
        while buffer.size >= 512:
            samples = buffer[:512]
            buffer = buffer[512:]
            chunk_capture = _capture_prefix(capture_buffer, 512)
            assert chunk_capture and sum(c.normalized_sample_count for c in chunk_capture) == 512
            events = vad.process_owned_chunk(samples, chunk_capture)
            consumed += 512
            observations.append(
                {
                    "frontier": consumed,
                    "speech": vad.last_observation_was_speech,
                    "events": [type(e).__name__ for e in events],
                    "input_frontier_ms": in_count / rate * 1000,
                }
            )
        pending.append(in_count / rate * 1000 - consumed / 16000 * 1000)
    tail, mapped, discarded = mapper.finish(orderly=True)
    assert not discarded and (
        mapped is None or mapped.normalized_start_sample == consumed + buffer.size
    )
    if mapped is not None:
        capture_buffer.append(mapped)
    assert sum(c.normalized_sample_count for c in capture_buffer) == buffer.size + tail.size
    normalized_total = consumed + buffer.size + tail.size
    assert normalized_total == 3 * 16000
    # No input after finite end; real residual must remain mapped, not padded into an audio claim.
    aborted = CaptureMappedStreamingResampler(rate)
    first_sample = signal[:block]
    capture = AudioCaptureSpan(0, 0, rate, 0, len(first_sample), 0, len(first_sample) / rate)
    aborted.process(first_sample, capture)
    abort_tail, _, abort_discarded = aborted.finish(orderly=False)
    assert abort_tail.size == 0
    disrupted = CaptureMappedStreamingResampler(rate)
    disrupted.process(first_sample, capture)
    next_capture = AudioCaptureSpan(
        1,
        1,
        rate,
        0,
        len(first_sample),
        1,
        1 + len(first_sample) / rate,
        discontinuity_before=AudioCaptureDiscontinuity("unknown_loss", 1),
    )
    _, _, lost = disrupted.process(first_sample, next_capture)
    assert all(item.capture_epoch == 0 for item in lost)
    stereo = MonoFirstStreamingResampler(rate, input_channels=2)
    mono = MonoFirstStreamingResampler(rate)
    stereo_parts = []
    mono_parts = []
    for start in range(0, signal.size, block):
        chunk = signal[start : start + block]
        stereo_parts.append(stereo.resample_chunk(np.column_stack((chunk, np.zeros_like(chunk)))))
        mono_parts.append(mono.resample_chunk(chunk * 0.5))
    stereo_parts.append(stereo.flush())
    mono_parts.append(mono.flush())
    np.testing.assert_allclose(np.concatenate(stereo_parts), np.concatenate(mono_parts), atol=1e-6)
    return {
        "rate": rate,
        "block": block,
        "first_output_input_frontier_ms": first,
        "actual_mapped_vad_frontier_age_ms": dist(pending),
        "vad_observation_count": len(observations),
        "vad_speech_observation_frontiers": [o["frontier"] for o in observations if o["speech"]][
            :5
        ],
        "vad_event_frontiers": [[o["frontier"], e] for o in observations for e in o["events"]],
        "normalized_before_flush": consumed + buffer.size,
        "finite_flush_tail": int(tail.size),
        "normalized_total": normalized_total,
        "mapped_span_count": len(spans),
        "abort_flush_samples": int(abort_tail.size),
        "abort_discard_count": len(abort_discarded),
        "unknown_gap_discard_count": len(lost),
        "mono_first_stereo_equivalent_to_mean": True,
        "fixture": "Generated voiced-tone envelope, NOT representative human speech; amplitude-engine interface, NOT Silero/ASR quality.",
    }


def main():
    assert version("soxr") == "1.1.0", "Requires pinned python-soxr 1.1.0"
    result = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {name: version(name) for name in ("soxr", "numpy")},
        "libsoxr": soxr.__libsoxr_version__,
        "grid": [],
        "mapped_path": [],
    }
    for rate, block in ((48000, 480), (48000, 1024), (44100, 441), (44100, 1024), (16000, 160)):
        for q in (("MQ", "LQ") if rate != 16000 else ("bypass",)):
            result["grid"].append(grid(rate, block, q))
        result["mapped_path"].append(mapped_path(rate, block))
    path = Path(__file__).with_name("e_results.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "grid": [
                    [
                        x["rate"],
                        x["block"],
                        x["quality"],
                        x["pending_ms"]["p50"],
                        x["flush_tail_samples"],
                    ]
                    for x in result["grid"]
                ],
                "mapped_check": [
                    [x["rate"], x["block"], x["normalized_total"]] for x in result["mapped_path"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
